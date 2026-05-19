//! Parsers for SEC EDGAR file formats.
//!
//! Each submodule is a pure byte-stream → typed-record transformer.
//! No I/O, no async, no network. This means every parser is trivially
//! testable against a fixture file and reusable from any orchestrator.

pub mod f13f;
pub mod form4;
pub mod fsnds;
pub mod idx;
pub mod submissions;
