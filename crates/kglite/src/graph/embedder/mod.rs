//! Embedder trait — pluggable text-embedding backend.
//!
//! kglite supports semantic search via `text_score()` in Cypher
//! queries. To embed query strings at lookup time the graph holds
//! an optional embedder. This trait abstracts over the available
//! backends:
//!
//! - [`fastembed::FastEmbedAdapter`] (gated on the `fastembed`
//!   Cargo feature) — Rust-native ONNX inference via
//!   `fastembed-rs`.
//!
//! - `PyEmbedderAdapter` (in kglite-py) — wraps a user-provided
//!   Python class. Used by the Python API path
//!   (`g.set_embedder(my_python_obj)`). PyO3-only; not visible
//!   here.
//!
//! Both implement [`Embedder`]; downstream consumers (the Cypher
//! engine's text-score rewrite, the `embed_texts` / `search_text`
//! pymethods) call through the trait without caring which backend
//! they got.

#[cfg(feature = "fastembed")]
pub mod fastembed;

/// Pluggable text-embedding backend. Implementations must be
/// `Send + Sync` because the `KnowledgeGraph` is freely cloned
/// across threads (its `embedder` field is an `Arc<dyn Embedder>`).
pub trait Embedder: Send + Sync {
    /// Embedding vector dimensionality (e.g. 1024 for BAAI/bge-m3,
    /// 384 for all-MiniLM-L6-v2). Used at `set_embeddings()` time
    /// to validate that user-supplied vectors match what the
    /// embedder produces.
    fn dimension(&self) -> usize;

    /// Embed a batch of texts into vectors. The returned outer Vec
    /// has the same length as the input slice (one vector per
    /// text); each inner Vec has length [`Self::dimension`].
    fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, String>;

    /// Stable identifier of the model producing these vectors (e.g.
    /// `"BAAI/bge-m3"`, `"sentence-transformers/all-MiniLM-L6-v2"`).
    /// Stamped onto the embedding store so a model swap is detectable
    /// after a save/load without external bookkeeping, and surfaced via
    /// `embedding_info()`. Default `None` — a backend that can't name
    /// its model simply leaves the store's `model_id` unset.
    fn model_id(&self) -> Option<String> {
        None
    }

    /// Optional lifecycle hook. Called by `embed_texts` /
    /// `search_text` before each embedding pass so the
    /// implementation can lazily materialise heavy resources
    /// (model weights, ONNX session, etc.) Default: no-op.
    fn load(&self) -> Result<(), String> {
        Ok(())
    }

    /// Optional lifecycle hook. Called after each embedding pass —
    /// implementations typically use this to schedule a cooldown
    /// timer that frees resources after some idle period. Default:
    /// no-op. Errors are silently ignored by callers since this is
    /// cleanup.
    fn unload(&self) {}
}
