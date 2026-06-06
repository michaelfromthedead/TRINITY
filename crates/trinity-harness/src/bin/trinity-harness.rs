//! Trinity Harness CLI binary.

use trinity_harness::cli::execute_command;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect(); // skip binary name
    let result = execute_command(&args);

    if result.success {
        println!("{}", result.message);
        std::process::exit(0);
    } else {
        eprintln!("Error: {}", result.message);
        std::process::exit(1);
    }
}
