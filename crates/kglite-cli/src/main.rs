//! Standalone, libpython-free frontend over the shared CLI library.

fn main() -> std::process::ExitCode {
    let result = kglite_cli::run(std::env::args_os());
    kglite_cli::exit_code(result)
}
