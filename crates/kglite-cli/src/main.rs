//! Standalone, libpython-free frontend over the shared CLI library.

fn main() -> std::process::ExitCode {
    match kglite_cli::run(std::env::args_os()) {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) if kglite_cli::is_reported_agent_failure(&error) => {
            std::process::ExitCode::FAILURE
        }
        Err(error) => {
            eprintln!("Error: {error:#}");
            std::process::ExitCode::FAILURE
        }
    }
}
