//! Pure bzip2-rs throughput microbench. Decompresses `<path>` to a
//! `/dev/null` sink with **zero parsing or scanning overhead** — tells
//! us the raw ceiling we're up against vs. the loader's observed rate.
//!
//! Usage:
//!     cargo run --bin bz2_bench --release -- <path-to-bz2> [preread-mb] [decompressed-cap-mb]
//!
//!     preread-mb       — bzip2-rs `max_preread_len` in MB (default 256, matches
//!                        loader's current `DEFAULT_BUDGET_BYTES`)
//!     decompressed-cap-mb — stop after this many MB of decompressed output
//!                          (default: read the whole file)
//!
//! We call `bzip2_rs::ParallelDecoderReader` directly instead of going
//! through `parallel_bz2::open` so we sidestep the stream-boundary
//! pre-scan — that's a separate concern, and we want the steady-state
//! number, not startup time.
use std::fs::File;
use std::io::{self, BufReader, Read, Write};
use std::path::Path;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: bz2_bench <path-to-bz2> [preread-mb=256] [decompressed-cap-mb=full]");
        std::process::exit(2);
    }
    let path = Path::new(&args[1]);
    let preread_mb: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(256);
    let preread_bytes = preread_mb * 1024 * 1024;

    let cap_bytes: Option<u64> = args
        .get(3)
        .and_then(|s| s.parse::<u64>().ok())
        .map(|mb| mb * 1024 * 1024);

    let file_size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);

    eprintln!(
        "bz2_bench: {} ({:.2} GB compressed)",
        path.display(),
        file_size as f64 / 1e9,
    );
    eprintln!("  preread:  {} MB ({}B)", preread_mb, preread_bytes,);
    if let Some(c) = cap_bytes {
        eprintln!("  cap:      {:.2} GB decompressed", c as f64 / 1e9);
    }

    let file = File::open(path).expect("open file");
    let reader = BufReader::with_capacity(8 * 1024 * 1024, file);
    let mut decoder =
        bzip2_rs::ParallelDecoderReader::new(reader, bzip2_rs::RayonThreadPool, preread_bytes);

    let start = Instant::now();
    let mut buf = vec![0u8; 8 * 1024 * 1024];
    let mut total: u64 = 0;
    let mut last_log = start;

    loop {
        let n = match decoder.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => n,
            Err(e) => {
                eprintln!("read error after {} bytes: {}", total, e);
                std::process::exit(1);
            }
        };
        total += n as u64;
        if let Some(cap) = cap_bytes {
            if total >= cap {
                break;
            }
        }
        if last_log.elapsed().as_secs_f64() >= 2.0 {
            let elapsed = start.elapsed().as_secs_f64();
            let mb_s = (total as f64 / 1e6) / elapsed;
            eprintln!(
                "  {:.2} GB decompressed in {:.1}s = {:.1} MB/s",
                total as f64 / 1e9,
                elapsed,
                mb_s,
            );
            let _ = io::stderr().flush();
            last_log = Instant::now();
        }
    }

    let elapsed = start.elapsed().as_secs_f64();
    let mb_decompressed_per_sec = (total as f64 / 1e6) / elapsed;
    let triples_per_sec = mb_decompressed_per_sec / 80.0; // ~80 bytes/triple

    println!();
    println!("=== BZ2 BENCHMARK ===");
    println!("  preread:         {} MB", preread_mb);
    println!(
        "  decompressed:    {:.2} GB ({:.0} bytes)",
        total as f64 / 1e9,
        total as f64
    );
    println!("  wall time:       {:.2} s", elapsed);
    println!(
        "  decompress rate: {:.1} MB/s decompressed",
        mb_decompressed_per_sec
    );
    println!(
        "  triple-rate eq:  {:.2} M tri/s (assuming ~80 bytes/triple)",
        triples_per_sec
    );
}
