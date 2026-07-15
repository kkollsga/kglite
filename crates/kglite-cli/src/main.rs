//! Standalone, libpython-free frontend over the shared CLI library.

fn main() -> anyhow::Result<()> {
    kglite_cli::run(std::env::args_os())
}
