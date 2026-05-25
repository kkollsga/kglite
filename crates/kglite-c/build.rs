//! Build-time cbindgen invocation — regenerates `include/kglite.h`
//! from the crate's #[no_mangle] extern "C" surface on every build.
//! CI diffs the committed header against fresh-cbindgen output to
//! catch hand-edits.

use std::env;
use std::path::PathBuf;

fn main() {
    // Re-run when any source file or config changes. cargo doesn't
    // auto-detect that cbindgen's output depends on the whole crate;
    // we have to enumerate the source dir.
    println!("cargo:rerun-if-changed=src");
    println!("cargo:rerun-if-changed=cbindgen.toml");
    println!("cargo:rerun-if-changed=build.rs");

    let crate_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let header_path = crate_dir.join("include").join("kglite.h");

    // Best-effort: if cbindgen fails (e.g. on an exotic target where
    // the crate is being built but the header isn't needed), don't
    // fail the build. The committed header in include/kglite.h is
    // still the source of truth for consumers; this regen step is
    // for keeping it fresh on the maintainer's machine.
    let config = match cbindgen::Config::from_file(crate_dir.join("cbindgen.toml")) {
        Ok(cfg) => cfg,
        Err(e) => {
            println!("cargo:warning=cbindgen config load failed: {e}");
            return;
        }
    };

    match cbindgen::Builder::new()
        .with_crate(crate_dir.clone())
        .with_config(config)
        .generate()
    {
        Ok(bindings) => {
            // Ensure include/ exists.
            let _ = std::fs::create_dir_all(crate_dir.join("include"));
            bindings.write_to_file(&header_path);
        }
        Err(e) => {
            println!("cargo:warning=cbindgen generate failed: {e}");
        }
    }
}
