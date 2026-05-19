//! Parsers for SEC EDGAR file formats.
//!
//! Each submodule is a pure byte-stream → typed-record transformer.
//! No I/O, no async, no network. This means every parser is trivially
//! testable against a fixture file and reusable from any orchestrator.

pub mod eightk;
pub mod exhibit21;
pub mod f13f;
pub mod form144;
pub mod form4;
pub mod fsnds;
pub mod idx;
pub mod ownership_table;
pub mod sc13d;
pub mod submissions;
