//! Embedding-backend selection from `extensions.embedder`, and the
//! wheel-supplied Python factory type the standalone binary lacks.

use std::sync::Arc;

use anyhow::Result;
use mcp_methods::server::Manifest;

/// Builds a graph embedder from the manifest's `extensions.embedder` config
/// (passed as a JSON string), on demand.
///
/// This is the seam that lets the **pip-hosted** server use *any* Python
/// embedding library (`extensions.embedder.library: sentence-transformers`,
/// `fastembed`, or a `factory:` escape) without the libpython-free library
/// knowing anything about Python: the kglite-py wrapper hands the config JSON
/// to a Python factory (`kglite._mcp_embed`) which picks the library, builds
/// the model, and wraps it in a `PyEmbedderAdapter` (GIL re-acquired only for
/// the embed call). The standalone cargo binary passes no factory, so a Python
/// library errors there with a clear message; it uses `library: fastembed-rs`
/// (the Rust `FastEmbedAdapter`) instead.
///
/// The argument is the whole `extensions.embedder` JSON object, so new fields
/// (library / model / factory / kwargs / …) flow through to Python without any
/// Rust change. `Send` because `run_with_embedder_factory` may move it into the
/// tokio runtime's future.
pub type PyEmbedderFactory =
    Box<dyn Fn(&str) -> Result<Arc<dyn kglite::api::Embedder>, String> + Send>;

/// Read `manifest.extensions.embedder.{library, model, …}` and build the
/// corresponding [`kglite::api::Embedder`]. Returns `Ok(None)` when no
/// `embedder:` is declared, `Err` on validation failures.
///
/// The `library` field names the embedding engine; the host (Rust vs Python)
/// is inferred from it:
/// - `fastembed-rs` — the Rust-native fastembed-rs adapter (cargo
///   `--features fastembed`; the only option on the standalone binary).
/// - any other value, or a `factory:` escape — a Python embedding library
///   (`fastembed`, `sentence-transformers`, …) built by `py_embedder_factory`
///   (supplied only by the pip-hosted server). The whole config object is
///   handed to Python as JSON, so the library set + its options live entirely
///   on the Python side (`kglite._mcp_embed`) — adding a library never touches
///   this function.
pub(crate) fn build_embedder_from_manifest(
    manifest: &Manifest,
    py_embedder_factory: Option<&PyEmbedderFactory>,
) -> Result<Option<Arc<dyn kglite::api::Embedder>>> {
    let Some(raw) = manifest.extensions.get("embedder") else {
        return Ok(None);
    };
    if !manifest.trust.allow_embedder {
        anyhow::bail!(
            "extensions.embedder is disabled unless the manifest explicitly sets \
             trust.allow_embedder: true"
        );
    }
    let obj = raw
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("extensions.embedder must be a mapping (got: {raw:?})"))?;
    // `fastembed-rs` is the only Rust-hosted engine; everything else (and any
    // `factory:`) is a Python library hosted by the wheel. Default to a Python
    // library so the common pip case needs only `library: + model:`.
    let library = obj.get("library").and_then(|v| v.as_str());
    let is_rust = library == Some("fastembed-rs");

    if is_rust {
        let model = obj.get("model").and_then(|v| v.as_str()).ok_or_else(|| {
            anyhow::anyhow!("extensions.embedder.model is required for library: fastembed-rs")
        })?;
        return build_rust_embedder(model);
    }

    // Python-hosted: hand the whole config object to the Python factory, which
    // picks the library, builds the model, and wraps it. The cargo binary
    // supplies no factory.
    let factory = py_embedder_factory.ok_or_else(|| {
        let lib = library.unwrap_or("<a Python library>");
        anyhow::anyhow!(
            "extensions.embedder.library = {lib:?} is a Python embedding library, but the \
             standalone `cargo install kglite-mcp-server` binary has no Python interpreter to \
             host it. Either run the server from the kglite wheel (`pip install kglite`, then \
             `pip install {lib}`), or use `library: fastembed-rs` with `cargo install \
             kglite-mcp-server --features fastembed`."
        )
    })?;
    let config_json = serde_json::to_string(raw)
        .map_err(|e| anyhow::anyhow!("serializing extensions.embedder failed: {e}"))?;
    let embedder = factory(&config_json)
        .map_err(|e| anyhow::anyhow!("python embedder construction failed: {e}"))?;
    tracing::info!(library = ?library, "registered python embedder");
    Ok(Some(embedder))
}

/// Build the Rust-native fastembed-rs embedder (`library: fastembed-rs`).
/// Gated on the `fastembed` cargo feature; the default build errors with a
/// rebuild hint (the feature is off by default because ort-sys has a flaky
/// upstream binary download).
#[cfg(feature = "fastembed")]
pub(crate) fn build_rust_embedder(model: &str) -> Result<Option<Arc<dyn kglite::api::Embedder>>> {
    let adapter = kglite::api::FastEmbedAdapter::new(model)
        .map_err(|e| anyhow::anyhow!("fastembed-rs init failed: {e}"))?;
    tracing::info!(model, "registered Rust-native (fastembed-rs) embedder");
    Ok(Some(Arc::new(adapter)))
}

#[cfg(not(feature = "fastembed"))]
pub(crate) fn build_rust_embedder(_model: &str) -> Result<Option<Arc<dyn kglite::api::Embedder>>> {
    anyhow::bail!(
        "extensions.embedder.library = \"fastembed-rs\" requires this binary to be built with \
         the `fastembed` feature: `cargo install kglite-mcp-server --features fastembed`. The \
         default build excludes it because its ort-sys dependency has a flaky upstream binary \
         download. (If you are running the pip wheel, use a Python library instead — e.g. \
         `library: sentence-transformers` with `pip install sentence-transformers`.)"
    )
}

#[cfg(test)]
mod embedder_trust_tests {
    use super::*;

    use std::sync::Arc;

    use mcp_methods::server::Manifest;

    use std::fs;
    use std::sync::atomic::{AtomicBool, Ordering};

    fn load_embedder_manifest(allow_embedder: bool) -> (tempfile::TempDir, Manifest) {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("mcp.yaml");
        fs::write(
            &path,
            format!(
                "name: trust-test\ntrust:\n  allow_embedder: {allow_embedder}\n\
                 extensions:\n  embedder:\n    library: test\n    model: test\n"
            ),
        )
        .expect("write manifest");
        let manifest = mcp_methods::server::load_manifest(&path).expect("load manifest");
        (dir, manifest)
    }

    #[test]
    fn untrusted_embedder_never_invokes_factory() {
        let (_dir, manifest) = load_embedder_manifest(false);
        let called = Arc::new(AtomicBool::new(false));
        let called_by_factory = called.clone();
        let factory: PyEmbedderFactory = Box::new(move |_| {
            called_by_factory.store(true, Ordering::SeqCst);
            Err("factory sentinel".to_string())
        });

        let error = match build_embedder_from_manifest(&manifest, Some(&factory)) {
            Ok(_) => panic!("untrusted embedder must be rejected"),
            Err(error) => error,
        };

        assert!(error.to_string().contains("trust.allow_embedder: true"));
        assert!(!called.load(Ordering::SeqCst));
    }

    #[test]
    fn trusted_embedder_reaches_factory() {
        let (_dir, manifest) = load_embedder_manifest(true);
        let called = Arc::new(AtomicBool::new(false));
        let called_by_factory = called.clone();
        let factory: PyEmbedderFactory = Box::new(move |_| {
            called_by_factory.store(true, Ordering::SeqCst);
            Err("factory sentinel".to_string())
        });

        let error = match build_embedder_from_manifest(&manifest, Some(&factory)) {
            Ok(_) => panic!("sentinel factory must fail"),
            Err(error) => error,
        };

        assert!(error.to_string().contains("factory sentinel"));
        assert!(called.load(Ordering::SeqCst));
    }
}
