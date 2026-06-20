//! Extract orchestrator — turns immutable raw/ filings into a set of
//! info-row CSVs in processed/.
//!
//! ## Design
//!
//! The 0.9.46 rewrite reshapes processed/ around information types
//! (Purchase, Sale, Holding, Role, …) rather than filing types. The
//! same info type can be populated from multiple form sources — every
//! Form 4 lot's `shares_owned_after`, every Form 3 initial holding,
//! every DEF 14A ownership-table row, every SC 13D/G 5%+ snapshot,
//! and every 10-K Item 12 disclosure all land in `processed/holding.csv`,
//! distinguished by their `source_form` provenance column.
//!
//! See `kglite/datasets/sec/FEATURE_GOALPOST.md` for the full coverage
//! target.
//!
//! ## Layers
//!
//! - **`sinks`** owns the CSV writers (one per info-row table) and
//!   their headers. Extractors only see `&mut Sinks` — they can't
//!   forget to write provenance because the helper appends it.
//! - **`identity/`** handles the identity tables (company, person,
//!   security, manager). Companies come from submissions.zip + master.idx
//!   up front; the rest are populated incrementally as forms reference
//!   them, with per-run dedup sets to prevent duplicates.
//! - **`forms/`** has one module per SEC form family. Each exposes
//!   `extract(workdir, slice, sinks, prov_base) -> Result<FormReport>`.
//!   Modules for forms we haven't implemented yet are real files
//!   (with documented CSV headers) that return Ok(0) — depth can
//!   be added per-form without orchestrator churn.
//! - **`orchestrator::run_all`** walks raw/, calls identity setup,
//!   then dispatches each form family to its module.
//! - **`provenance::Provenance`** carries the 8-column footer.
//! - **`util`** holds path parsers and small format helpers.
//!
//! ## Public API
//!
//! `run_all(workdir, slice, force) -> Result<ExtractReport>` is the
//! single entry point. PyO3 layer (`src/sec.rs`) exposes exactly one
//! binding, `extract_all_py`, that calls into here.

pub mod forms;
pub mod identity;
pub mod orchestrator;
pub mod provenance;
pub mod sinks;
pub mod util;

pub use orchestrator::{run_all, ExtractReport};
