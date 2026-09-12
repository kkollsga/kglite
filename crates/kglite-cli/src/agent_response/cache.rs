//! Persistent storage for complete agent-response envelopes.
//!
//! The cache owns serialized JSON values only. It has no graph, query, working
//! directory, or presentation dependency, so loading a handle cannot replay an
//! operation. One lock and one flat entry directory enforce limits across all
//! workspace namespaces.

use std::ffi::{OsStr, OsString};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

const FORMAT_VERSION: u8 = 1;
const HANDLE_PREFIX: &str = "kgr_";
const ENTRY_SUFFIX: &str = ".json";
const TEMP_MARKER: &str = ".tmp-";
const DEFAULT_TTL: Duration = Duration::from_secs(10 * 60);
const DEFAULT_MAX_ENTRIES: usize = 32;
const DEFAULT_MAX_BYTES: u64 = 32 * 1024 * 1024;

/// Provenance and complete canonical envelope recovered from one opaque handle.
#[derive(Debug, PartialEq)]
pub(crate) struct LoadedResult {
    pub(crate) namespace: String,
    pub(crate) created_at_unix_seconds: u64,
    pub(crate) envelope: Value,
}

pub(crate) trait Clock: Clone + Send + Sync + 'static {
    fn unix_seconds(&self) -> Result<u64>;
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct SystemClock;

impl Clock for SystemClock {
    fn unix_seconds(&self) -> Result<u64> {
        Ok(SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .context("system clock is before the Unix epoch")?
            .as_secs())
    }
}

pub(crate) trait IdSource: Clone + Send + Sync + 'static {
    fn next_id(&self) -> Result<[u8; 16]>;
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct OsRandomIds;

impl IdSource for OsRandomIds {
    fn next_id(&self) -> Result<[u8; 16]> {
        let mut bytes = [0_u8; 16];
        getrandom::fill(&mut bytes)
            .map_err(|error| anyhow!("OS randomness unavailable: {error}"))?;
        Ok(bytes)
    }
}

#[derive(Clone, Copy, Debug)]
struct Limits {
    ttl: Duration,
    max_entries: usize,
    max_bytes: u64,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            ttl: DEFAULT_TTL,
            max_entries: DEFAULT_MAX_ENTRIES,
            max_bytes: DEFAULT_MAX_BYTES,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct Record {
    format_version: u8,
    handle: String,
    namespace: String,
    created_at_unix_seconds: u64,
    creation_sequence: u64,
    envelope: Value,
}

#[derive(Debug)]
struct Entry {
    path: PathBuf,
    creation_sequence: u64,
    bytes: u64,
}

/// Globally bounded per-user cache. Workspace namespace is record provenance,
/// not lookup context: `response expand HANDLE` needs only this root and handle.
#[derive(Clone)]
pub(crate) struct ResultCache<C = SystemClock, I = OsRandomIds> {
    root: PathBuf,
    limits: Limits,
    clock: C,
    ids: I,
    #[cfg(test)]
    fault: TestFault,
}

impl ResultCache<SystemClock, OsRandomIds> {
    pub(crate) fn new(root: PathBuf) -> Self {
        Self::with_parts(root, Limits::default(), SystemClock, OsRandomIds)
    }
}

impl<C: Clock, I: IdSource> ResultCache<C, I> {
    fn with_parts(root: PathBuf, limits: Limits, clock: C, ids: I) -> Self {
        Self {
            root,
            limits,
            clock,
            ids,
            #[cfg(test)]
            fault: TestFault::None,
        }
    }

    /// Store the complete canonical envelope and return an opaque global handle.
    pub(crate) fn store(&self, namespace: &str, envelope: Value) -> Result<String> {
        if namespace.is_empty() {
            bail!("retained-result namespace must not be empty");
        }
        let _lock = self.lock_root()?;
        let now = self.clock.unix_seconds()?;
        self.cleanup_locked(now)?;
        let sequence = self.next_sequence_locked()?;

        for _ in 0..8 {
            let handle = format!("{HANDLE_PREFIX}{}", hex(&self.ids.next_id()?));
            let final_path = self.entry_path(&handle);
            if fs::symlink_metadata(&final_path).is_ok() {
                continue;
            }
            let record = Record {
                format_version: FORMAT_VERSION,
                handle: handle.clone(),
                namespace: namespace.to_owned(),
                created_at_unix_seconds: now,
                creation_sequence: sequence,
                envelope: envelope.clone(),
            };
            let encoded = serde_json::to_vec(&record).context("serialize retained result")?;
            let incoming_bytes =
                u64::try_from(encoded.len()).context("retained result is too large")?;
            if incoming_bytes > self.limits.max_bytes {
                bail!("retained result exceeds the global cache byte limit");
            }

            // The in-memory encoding fixes the exact candidate size. Evicting
            // first leaves room for the temp file itself, so entry metadata and
            // incomplete bytes remain inside both strict global caps. Capacity
            // eviction is best effort, not transactional: a later publication
            // failure does not restore already-evicted old handles.
            self.evict_for_locked(incoming_bytes, 1)?;
            let temp_path = self.temp_path(&handle);
            let publication = self.publish(&temp_path, &final_path, &encoded);
            if let Err(error) = publication {
                if fs::symlink_metadata(&temp_path).is_ok() {
                    let _ = remove_cache_object(&temp_path);
                }
                return Err(error);
            }
            return Ok(handle);
        }
        bail!("could not allocate a collision-free retained-result handle")
    }

    /// Load a retained result without a graph lookup or replay.
    pub(crate) fn load(&self, handle: &str) -> Result<Option<LoadedResult>> {
        validate_handle(handle)?;
        let _lock = self.lock_root()?;
        let now = self.clock.unix_seconds()?;
        self.cleanup_locked(now)?;
        let path = self.entry_path(handle);
        let Some(record) = read_valid_record(&path)? else {
            return Ok(None);
        };
        if expired(record.created_at_unix_seconds, now, self.limits.ttl) {
            remove_regular(&path)?;
            return Ok(None);
        }
        Ok(Some(LoadedResult {
            namespace: record.namespace,
            created_at_unix_seconds: record.created_at_unix_seconds,
            envelope: record.envelope,
        }))
    }

    /// Clear the complete per-user cache across workspace namespaces.
    pub(crate) fn purge_all(&self) -> Result<()> {
        let _lock = self.lock_root()?;
        for item in fs::read_dir(self.entries_dir())? {
            remove_cache_object(&item?.path())?;
        }
        Ok(())
    }

    fn publish(&self, temp_path: &Path, final_path: &Path, bytes: &[u8]) -> Result<()> {
        #[cfg(test)]
        if self.fault == TestFault::BeforeWrite {
            bail!("injected cache write failure");
        }
        let mut file = private_open(temp_path, false)?;
        #[cfg(test)]
        if self.fault == TestFault::AfterPrefix {
            file.write_all(&bytes[..bytes.len().min(8)])?;
            file.sync_all()?;
            bail!("injected cache partial-write failure");
        }
        file.write_all(bytes)?;
        file.sync_all()?;
        drop(file);
        #[cfg(test)]
        if self.fault == TestFault::BeforeRename {
            bail!("injected cache rename failure");
        }
        fs::rename(temp_path, final_path).with_context(|| {
            format!(
                "atomically publish retained result {}",
                final_path.display()
            )
        })
    }

    fn lock_root(&self) -> Result<File> {
        ensure_private_dir(&self.root)?;
        ensure_private_dir(&self.entries_dir())?;
        let lock_path = self.root.join("lock");
        reject_symlink_or_non_file_if_present(&lock_path)?;
        let file = private_open(&lock_path, true)?;
        file.lock()
            .with_context(|| format!("lock result cache {}", self.root.display()))?;
        Ok(file)
    }

    fn cleanup_locked(&self, now: u64) -> Result<()> {
        for item in fs::read_dir(self.entries_dir())? {
            let path = item?.path();
            let name = path.file_name().and_then(OsStr::to_str).unwrap_or("");
            if name.contains(TEMP_MARKER) || !name.ends_with(ENTRY_SUFFIX) {
                remove_cache_object(&path)?;
                continue;
            }
            match read_valid_record(&path) {
                Ok(Some(record))
                    if expired(record.created_at_unix_seconds, now, self.limits.ttl) =>
                {
                    remove_regular(&path)?;
                }
                Ok(Some(_)) => {}
                Ok(None) | Err(_) => remove_cache_object(&path)?,
            }
        }
        self.evict_for_locked(0, 0)
    }

    fn evict_for_locked(&self, incoming_bytes: u64, incoming_entries: usize) -> Result<()> {
        let mut entries = scan_entries(&self.entries_dir())?;
        entries.sort_by_key(|entry| entry.creation_sequence);
        let mut bytes: u64 = entries.iter().map(|entry| entry.bytes).sum();
        let mut count = entries.len();
        while count.saturating_add(incoming_entries) > self.limits.max_entries
            || bytes.saturating_add(incoming_bytes) > self.limits.max_bytes
        {
            let Some(oldest) = entries.first() else {
                bail!("cache limits cannot admit retained result");
            };
            remove_regular(&oldest.path)?;
            bytes = bytes.saturating_sub(oldest.bytes);
            count -= 1;
            entries.remove(0);
        }
        Ok(())
    }

    fn next_sequence_locked(&self) -> Result<u64> {
        match scan_entries(&self.entries_dir())?
            .into_iter()
            .map(|entry| entry.creation_sequence)
            .max()
        {
            None => Ok(0),
            Some(value) => value.checked_add(1).ok_or_else(|| {
                anyhow!("retained-result creation sequence exhausted; purge the cache before storing more results")
            }),
        }
    }

    fn entries_dir(&self) -> PathBuf {
        self.root.join("entries")
    }

    fn entry_path(&self, handle: &str) -> PathBuf {
        self.entries_dir().join(entry_filename(handle))
    }

    fn temp_path(&self, handle: &str) -> PathBuf {
        self.entries_dir()
            .join(format!(".{handle}{TEMP_MARKER}{}", std::process::id()))
    }
}

fn scan_entries(dir: &Path) -> Result<Vec<Entry>> {
    let mut out = Vec::new();
    for item in fs::read_dir(dir)? {
        let path = item?.path();
        let name = path.file_name().and_then(OsStr::to_str).unwrap_or("");
        if !name.ends_with(ENTRY_SUFFIX) || name.contains(TEMP_MARKER) {
            continue;
        }
        let Some(record) = read_valid_record(&path)? else {
            continue;
        };
        out.push(Entry {
            bytes: fs::symlink_metadata(&path)?.len(),
            path,
            creation_sequence: record.creation_sequence,
        });
    }
    Ok(out)
}

fn read_valid_record(path: &Path) -> Result<Option<Record>> {
    let metadata = match fs::symlink_metadata(path) {
        Ok(value) => value,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error.into()),
    };
    if metadata.file_type().is_symlink() || !metadata.file_type().is_file() {
        bail!("refusing non-regular cache entry {}", path.display());
    }
    validate_private_mode(&metadata, path)?;
    let mut bytes = Vec::with_capacity(usize::try_from(metadata.len()).unwrap_or(0));
    File::open(path)?.read_to_end(&mut bytes)?;
    let record: Record = serde_json::from_slice(&bytes).context("decode retained result")?;
    if record.format_version != FORMAT_VERSION
        || entry_filename(&record.handle) != path.file_name().unwrap_or_default()
    {
        bail!("invalid retained-result record {}", path.display());
    }
    validate_handle(&record.handle)?;
    Ok(Some(record))
}

fn validate_handle(handle: &str) -> Result<()> {
    let tail = handle
        .strip_prefix(HANDLE_PREFIX)
        .ok_or_else(|| anyhow!("invalid result handle"))?;
    if tail.len() != 32
        || !tail
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        bail!("invalid result handle");
    }
    Ok(())
}

fn entry_filename(handle: &str) -> OsString {
    format!("{handle}{ENTRY_SUFFIX}").into()
}

fn expired(created: u64, now: u64, ttl: Duration) -> bool {
    now.saturating_sub(created) >= ttl.as_secs()
}

fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        out.push(DIGITS[(byte >> 4) as usize] as char);
        out.push(DIGITS[(byte & 0x0f) as usize] as char);
    }
    out
}

fn ensure_private_dir(path: &Path) -> Result<()> {
    match fs::symlink_metadata(path) {
        Ok(metadata) => {
            if metadata.file_type().is_symlink() || !metadata.is_dir() {
                bail!("refusing unsafe cache directory {}", path.display());
            }
            validate_private_mode(&metadata, path)
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => create_private_dir(path),
        Err(error) => Err(error.into()),
    }
}

fn reject_symlink_or_non_file_if_present(path: &Path) -> Result<()> {
    if let Ok(metadata) = fs::symlink_metadata(path) {
        if metadata.file_type().is_symlink() || !metadata.is_file() {
            bail!("refusing unsafe cache lock {}", path.display());
        }
        validate_private_mode(&metadata, path)?;
    }
    Ok(())
}

#[cfg(unix)]
fn private_open(path: &Path, allow_existing: bool) -> Result<File> {
    use std::os::unix::fs::OpenOptionsExt;
    let mut options = OpenOptions::new();
    options.read(true).write(true).create(true).mode(0o600);
    if !allow_existing {
        options.create_new(true);
    }
    Ok(options.open(path)?)
}

#[cfg(not(unix))]
fn private_open(path: &Path, allow_existing: bool) -> Result<File> {
    let mut options = OpenOptions::new();
    options.read(true).write(true).create(true);
    if !allow_existing {
        options.create_new(true);
    }
    Ok(options.open(path)?)
}

#[cfg(unix)]
fn create_private_dir(path: &Path) -> Result<()> {
    use std::os::unix::fs::DirBuilderExt;
    let mut builder = fs::DirBuilder::new();
    builder.recursive(true).mode(0o700);
    builder.create(path)?;
    let metadata = fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        bail!("refusing unsafe cache directory {}", path.display());
    }
    validate_private_mode(&metadata, path)
}

#[cfg(not(unix))]
fn create_private_dir(path: &Path) -> Result<()> {
    fs::create_dir_all(path)?;
    let metadata = fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        bail!("refusing unsafe cache directory {}", path.display());
    }
    Ok(())
}

#[cfg(unix)]
fn validate_private_mode(metadata: &fs::Metadata, path: &Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    if metadata.permissions().mode() & 0o077 != 0 {
        bail!(
            "cache path is accessible by another user: {}",
            path.display()
        );
    }
    Ok(())
}

#[cfg(not(unix))]
fn validate_private_mode(_metadata: &fs::Metadata, _path: &Path) -> Result<()> {
    Ok(())
}

fn remove_regular(path: &Path) -> Result<()> {
    let metadata = fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        bail!(
            "refusing to remove non-regular cache entry {}",
            path.display()
        );
    }
    fs::remove_file(path)?;
    Ok(())
}

fn remove_cache_object(path: &Path) -> Result<()> {
    let metadata = fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() || metadata.is_file() {
        fs::remove_file(path)?;
        return Ok(());
    }
    bail!("refusing unknown cache object {}", path.display())
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
enum TestFault {
    #[default]
    None,
    BeforeWrite,
    AfterPrefix,
    BeforeRename,
}

#[cfg(test)]
mod tests;
