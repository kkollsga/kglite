//! Pure-Rust Sodir FactMaps REST loader for KGLite knowledge graphs.
//!
//! The crate has no Python or PyO3 dependency. PyO3 bindings live in
//! the main `kglite` crate under `src/sodir.rs`; the Python-facing API
//! is `kglite.datasets.sodir.open(...)`.
//!
//! It mirrors the `kglite-sec` crate's layered architecture
//! (dependencies flow strictly one direction):
//!
//! ```text
//! lib (public API)
//!   ├── orchestrator  refresh + fetch_all — drives index/client/fetch  [A4]
//!   ├── blueprint     blueprint walk + deep-merge                       [A4]
//!   ├── preprocess    the 4 FK-derivation joins                         [A3]
//!   ├── index         sodir_index.json + two-tier cooldown              [A3]
//!   ├── fetch         paginate ArcGIS GeoJSON → CSV                     [A2]
//!   ├── client        ArcGIS REST client (rate limit + retry)           [A2]
//!   ├── geojson_wkt   GeoJSON → WKT, epoch-ms → ISO date                [A1]
//!   ├── layout        Workdir tiers + StorageMode                       [A1]
//!   ├── catalog       ~150 dataset stem → (url, layer_id)               [A1]
//!   └── error         SodirError                                        [A1]
//! ```

pub mod catalog;
pub mod client;
pub mod error;
pub mod fetch;
pub mod geojson_wkt;
pub mod layout;

pub use catalog::{is_known, kind_of, resolve, DataKind};
pub use client::ArcGISClient;
pub use error::{Result, SodirError};
pub use layout::{StorageMode, Workdir};
