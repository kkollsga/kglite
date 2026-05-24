//! [`FastEmbedAdapter`] — Rust-native [`Embedder`] backed by
//! `fastembed-rs` (ONNX runtime under the hood).
//!
//! Used by `kglite-mcp-server` to support `text_score()` semantic
//! search without embedding Python. Operators opt in via:
//!
//! ```yaml
//! extensions:
//!   embedder:
//!     backend: fastembed
//!     model: BAAI/bge-m3       # or any supported model name
//!     cooldown: 900             # seconds of idle → drop weights (0 = never)
//! ```
//!
//! The first `embed()` call downloads ONNX weights to `~/.cache/fastembed/`
//! (or the system cache dir). Subsequent calls hit the cache. `unload()`
//! drops the `TextEmbedding` instance, freeing ~1-2 GB of resident memory;
//! the next `load()` re-materialises from the cached weights (no network).

use std::sync::Mutex;

use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};

use super::Embedder;

/// Rust-native embedder. Maps a model name (e.g. "BAAI/bge-m3") to
/// one of fastembed's `EmbeddingModel` variants. Lazy-init on first
/// `load()` or `embed()` so construction is cheap.
pub struct FastEmbedAdapter {
    model: EmbeddingModel,
    dimension: usize,
    inner: Mutex<Option<TextEmbedding>>,
}

impl FastEmbedAdapter {
    /// Build an adapter for the given model name. Returns an error if
    /// the name doesn't match a known fastembed model. Does **not**
    /// download anything yet — that happens on the first `load()` /
    /// `embed()` call.
    pub fn new(model_name: &str) -> Result<Self, String> {
        let (model, dimension) = resolve_model(model_name)?;
        Ok(Self {
            model,
            dimension,
            inner: Mutex::new(None),
        })
    }

    fn ensure_loaded(&self) -> Result<(), String> {
        let mut guard = self
            .inner
            .lock()
            .map_err(|e| format!("lock poisoned: {e}"))?;
        if guard.is_none() {
            let opts = InitOptions::new(self.model.clone()).with_show_download_progress(false);
            let te = TextEmbedding::try_new(opts)
                .map_err(|e| format!("fastembed init failed for {:?}: {e}", self.model))?;
            *guard = Some(te);
        }
        Ok(())
    }
}

impl Embedder for FastEmbedAdapter {
    fn dimension(&self) -> usize {
        self.dimension
    }

    fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, String> {
        self.ensure_loaded()?;
        let mut guard = self
            .inner
            .lock()
            .map_err(|e| format!("lock poisoned: {e}"))?;
        let te = guard
            .as_mut()
            .ok_or_else(|| "embedder loaded but missing".to_string())?;
        let inputs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        te.embed(inputs, None)
            .map_err(|e| format!("fastembed embed() failed: {e}"))
    }

    fn load(&self) -> Result<(), String> {
        self.ensure_loaded()
    }

    fn unload(&self) {
        if let Ok(mut guard) = self.inner.lock() {
            *guard = None;
        }
    }
}

/// Map a model name (e.g. "BAAI/bge-m3") to a fastembed
/// [`EmbeddingModel`] + its dimension. Returns an error for names not
/// in fastembed's catalog.
fn resolve_model(name: &str) -> Result<(EmbeddingModel, usize), String> {
    // Match on the common HF-style names operators put in YAML. The
    // canonical set fastembed ships is much larger — extend this map
    // as users ask for specific models. We pre-resolve the dimension
    // here so `dimension()` is sync and doesn't need to load the
    // model first.
    let (model, dim) = match name {
        "BAAI/bge-m3" => (EmbeddingModel::BGEM3, 1024),
        "BAAI/bge-small-en-v1.5" | "bge-small-en-v1.5" => (EmbeddingModel::BGESmallENV15, 384),
        "BAAI/bge-base-en-v1.5" | "bge-base-en-v1.5" => (EmbeddingModel::BGEBaseENV15, 768),
        "BAAI/bge-large-en-v1.5" | "bge-large-en-v1.5" => (EmbeddingModel::BGELargeENV15, 1024),
        "sentence-transformers/all-MiniLM-L6-v2" | "all-MiniLM-L6-v2" => {
            (EmbeddingModel::AllMiniLML6V2, 384)
        }
        "intfloat/multilingual-e5-large" | "multilingual-e5-large" => {
            (EmbeddingModel::MultilingualE5Large, 1024)
        }
        "intfloat/multilingual-e5-base" | "multilingual-e5-base" => {
            (EmbeddingModel::MultilingualE5Base, 768)
        }
        other => {
            return Err(format!(
                "unsupported fastembed model name: {other:?}. \
                 Known: BAAI/bge-m3, BAAI/bge-small-en-v1.5, BAAI/bge-base-en-v1.5, \
                 BAAI/bge-large-en-v1.5, sentence-transformers/all-MiniLM-L6-v2, \
                 intfloat/multilingual-e5-large, intfloat/multilingual-e5-base."
            ))
        }
    };
    Ok((model, dim))
}
