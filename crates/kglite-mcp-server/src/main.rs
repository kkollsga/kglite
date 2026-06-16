//! `kglite-mcp-server` binary — thin frontend over the crate library.
//!
//! All server logic lives in `lib.rs::run` so the exact same server can
//! also be driven in-process from the `kglite` Python wheel's PyO3
//! wrapper (`pip install kglite` ships this server with no separate
//! wheel and no duplicated engine). This binary is the libpython-free
//! `cargo install kglite-mcp-server` path; it just forwards argv.

fn main() -> anyhow::Result<()> {
    kglite_mcp_server::run(std::env::args_os())
}
