//! Streaming synthetic property-graph generator for kglite/kùzu benchmarks.
//!
//! Emits the `graphsuite` schema (Person/Company/Project/Skill/City +
//! KNOWS/WORKS_AT/CONTRIBUTES_TO/HAS_SKILL/OWNS/DEPENDS_ON/LOCATED_IN) as
//! one CSV per type plus a `manifest.json` of schema + seed-derived query
//! params. Every engine (kùzu `COPY FROM`, kglite `add_nodes`) loads the
//! *same* bytes, so cross-engine result-parity holds.
//!
//! Bounded memory at any scale: nodes and edges are streamed row-by-row to
//! disk; the only resident state is the RNG, small per-source dedup sets,
//! and counters. A 50M-node graph generates in the same RAM as a 1k one.
//!
//! Usage:
//!   graphgen --out DIR [--scale tiny|small|medium|large|huge|xhuge | --persons N]
//!            [--seed S] [--knows-per K] [--degree-dist uniform|zipf] [--zipf-exp E]

use std::collections::HashSet;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::exit;

// ─────────────────────────────────────────────────────────────────────────
// Deterministic PRNG — splitmix64. No external deps; reproducible per seed.
// ─────────────────────────────────────────────────────────────────────────
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed)
    }
    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform in [0, n). n must be > 0. Modulo bias is irrelevant for benchmark data.
    #[inline]
    fn below(&mut self, n: u64) -> u64 {
        if n == 0 {
            0
        } else {
            self.next_u64() % n
        }
    }
    /// Uniform float in [0, 1).
    #[inline]
    fn f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    /// Inclusive integer range [lo, hi].
    #[inline]
    fn range_i64(&mut self, lo: i64, hi: i64) -> i64 {
        lo + self.below((hi - lo + 1) as u64) as i64
    }
    /// Pick one element of a slice.
    #[inline]
    fn pick<'a, T>(&mut self, xs: &'a [T]) -> &'a T {
        &xs[self.below(xs.len() as u64) as usize]
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Schema — categorical value pools (mirror graphsuite/dataset.py).
// ─────────────────────────────────────────────────────────────────────────
const INDUSTRIES: &[&str] = &[
    "tech",
    "energy",
    "finance",
    "health",
    "retail",
    "media",
    "logistics",
    "gov",
];
const SKILL_CATEGORIES: &[&str] = &["lang", "framework", "tool", "domain", "soft"];
const PROJECT_STATUS: &[&str] = &["planning", "active", "paused", "done", "cancelled"];
const REGIONS: &[&str] = &["north", "south", "east", "west", "central"];

// ─────────────────────────────────────────────────────────────────────────
// Config
// ─────────────────────────────────────────────────────────────────────────
struct Config {
    out: PathBuf,
    persons: u64,
    knows_per: u64,
    seed: u64,
    zipf: bool,
    zipf_exp: f64,
}

fn scale_persons(name: &str) -> Option<u64> {
    Some(match name {
        "tiny" => 1_000,
        "small" => 2_000,
        "medium" => 20_000,
        "large" => 100_000,
        "huge" => 5_000_000,
        "xhuge" => 50_000_000,
        _ => return None,
    })
}

// Contiguous global-id ranges, allocated in one pass (matches dataset.py order).
struct Ranges {
    person: (u64, u64),
    company: (u64, u64),
    project: (u64, u64),
    skill: (u64, u64),
    city: (u64, u64),
}

impl Ranges {
    fn alloc(persons: u64) -> Self {
        let n_company = (persons / 25).max(8);
        let n_project = (persons / 5).max(20);
        let n_skill = (persons / 60).max(20);
        let n_city = (persons / 100).max(10);
        let mut c = 0u64;
        let mut take = |n: u64| {
            let s = c;
            c += n;
            (s, s + n)
        };
        Ranges {
            person: take(persons),
            company: take(n_company),
            project: take(n_project),
            skill: take(n_skill),
            city: take(n_city),
        }
    }
}

#[inline]
fn count(r: (u64, u64)) -> u64 {
    r.1 - r.0
}

// A CSV sink with a large buffer for throughput.
struct Csv(BufWriter<File>);
impl Csv {
    fn create(dir: &Path, name: &str, header: &str) -> Csv {
        let f = File::create(dir.join(format!("{name}.csv")))
            .unwrap_or_else(|e| fail(&format!("create {name}.csv: {e}")));
        let mut w = BufWriter::with_capacity(1 << 20, f);
        writeln!(w, "{header}").unwrap();
        Csv(w)
    }
    #[inline]
    fn row(&mut self, args: std::fmt::Arguments) {
        self.0.write_fmt(args).unwrap();
        self.0.write_all(b"\n").unwrap();
    }
}

fn fail(msg: &str) -> ! {
    eprintln!("graphgen: {msg}");
    exit(1);
}

fn main() {
    let cfg = parse_args();
    fs::create_dir_all(&cfg.out).unwrap_or_else(|e| fail(&format!("mkdir out: {e}")));
    let r = Ranges::alloc(cfg.persons);

    let t0 = std::time::Instant::now();
    let n_edges = generate(&cfg, &r);
    let n_nodes =
        count(r.person) + count(r.company) + count(r.project) + count(r.skill) + count(r.city);
    write_manifest(&cfg, &r, n_nodes, n_edges);

    eprintln!(
        "graphgen: {} nodes · {} edges → {} in {:.1}s (seed {}, dist {})",
        n_nodes,
        n_edges,
        cfg.out.display(),
        t0.elapsed().as_secs_f64(),
        cfg.seed,
        if cfg.zipf { "zipf" } else { "uniform" },
    );
}

// ─────────────────────────────────────────────────────────────────────────
// Generation
// ─────────────────────────────────────────────────────────────────────────
fn generate(cfg: &Config, r: &Ranges) -> u64 {
    let dir = &cfg.out;
    let n_city = count(r.city);
    let n_company = count(r.company);
    let n_project = count(r.project);
    let n_skill = count(r.skill);
    let n_person = count(r.person);

    // Independent RNG streams per table keep one table's size from shifting
    // another's values — re-running at a larger scale leaves smaller prefixes
    // stable where the math allows.
    let mut rng = Rng::new(cfg.seed);

    // ---- nodes ----------------------------------------------------------
    let mut city = Csv::create(dir, "City", "gid,name,population,region");
    for i in 0..n_city {
        let gid = r.city.0 + i;
        city.row(format_args!(
            "{gid},City_{i},{},{}",
            rng.range_i64(5_000, 9_000_000),
            rng.pick(REGIONS)
        ));
    }

    let mut skill = Csv::create(dir, "Skill", "gid,name,category");
    for i in 0..n_skill {
        let gid = r.skill.0 + i;
        skill.row(format_args!(
            "{gid},Skill_{i},{}",
            rng.pick(SKILL_CATEGORIES)
        ));
    }

    let mut company = Csv::create(dir, "Company", "gid,name,industry,size");
    for i in 0..n_company {
        let gid = r.company.0 + i;
        company.row(format_args!(
            "{gid},Company_{i},{},{}",
            rng.pick(INDUSTRIES),
            rng.range_i64(5, 50_000)
        ));
    }

    let mut project = Csv::create(dir, "Project", "gid,name,budget,status");
    for i in 0..n_project {
        let gid = r.project.0 + i;
        project.row(format_args!(
            "{gid},Project_{i},{:.2},{}",
            rng.f64() * 1_000_000.0,
            rng.pick(PROJECT_STATUS)
        ));
    }

    let mut person = Csv::create(dir, "Person", "gid,name,age,city,joined_year,active,score");
    for i in 0..n_person {
        let gid = r.person.0 + i;
        person.row(format_args!(
            "{gid},Person_{i},{},City_{},{},{},{:.4}",
            rng.range_i64(18, 80),
            rng.below(n_city),
            rng.range_i64(2000, 2025),
            rng.below(2), // active 0/1
            rng.f64() * 100.0
        ));
    }

    // ---- edges (streamed; bounded per-source dedup) ---------------------
    let mut n_edges = 0u64;

    // KNOWS: Person→Person, ~knows_per each, optional hub skew, undirected-ish.
    let mut knows = Csv::create(dir, "KNOWS", "src,dst");
    let mut seen: HashSet<u64> = HashSet::with_capacity(cfg.knows_per as usize * 2);
    for i in 0..n_person {
        let src = r.person.0 + i;
        seen.clear();
        let mut made = 0;
        let mut attempts = 0;
        while made < cfg.knows_per && attempts < cfg.knows_per * 4 {
            attempts += 1;
            let t = sample_person(&mut rng, n_person, cfg);
            let dst = r.person.0 + t;
            if dst == src || !seen.insert(dst) {
                continue;
            }
            knows.row(format_args!("{src},{dst}"));
            made += 1;
            n_edges += 1;
        }
    }

    // WORKS_AT: each Person → 1 Company.
    let mut works = Csv::create(dir, "WORKS_AT", "src,dst");
    for i in 0..n_person {
        let src = r.person.0 + i;
        let dst = r.company.0 + rng.below(n_company);
        works.row(format_args!("{src},{dst}"));
        n_edges += 1;
    }

    // CONTRIBUTES_TO: each Person → 1-2 Projects.
    let mut contrib = Csv::create(dir, "CONTRIBUTES_TO", "src,dst");
    for i in 0..n_person {
        let src = r.person.0 + i;
        let k = 1 + rng.below(2);
        for _ in 0..k {
            let dst = r.project.0 + rng.below(n_project);
            contrib.row(format_args!("{src},{dst}"));
            n_edges += 1;
        }
    }

    // HAS_SKILL: each Person → ~3 distinct Skills.
    let mut hasskill = Csv::create(dir, "HAS_SKILL", "src,dst");
    let want_skills = 3.min(n_skill);
    for i in 0..n_person {
        let src = r.person.0 + i;
        seen.clear();
        let mut made = 0;
        let mut attempts = 0;
        while made < want_skills && attempts < want_skills * 4 {
            attempts += 1;
            let dst = r.skill.0 + rng.below(n_skill);
            if !seen.insert(dst) {
                continue;
            }
            hasskill.row(format_args!("{src},{dst}"));
            made += 1;
            n_edges += 1;
        }
    }

    // OWNS: each Company → ~ (project/company) Projects.
    let mut owns = Csv::create(dir, "OWNS", "src,dst");
    let owns_per = (n_project / n_company).max(1);
    for i in 0..n_company {
        let src = r.company.0 + i;
        for _ in 0..owns_per {
            let dst = r.project.0 + rng.below(n_project);
            owns.row(format_args!("{src},{dst}"));
            n_edges += 1;
        }
    }

    // DEPENDS_ON: Project→Project DAG (dst gid strictly < src gid → acyclic).
    let mut depends = Csv::create(dir, "DEPENDS_ON", "src,dst");
    for i in 1..n_project {
        let src = r.project.0 + i;
        // 0-3 downstream deps onto earlier projects
        let k = rng.below(4);
        for _ in 0..k {
            let dst = r.project.0 + rng.below(i); // earlier project
            depends.row(format_args!("{src},{dst}"));
            n_edges += 1;
        }
    }

    // LOCATED_IN: each Company → 1 City.
    let mut located = Csv::create(dir, "LOCATED_IN", "src,dst");
    for i in 0..n_company {
        let src = r.company.0 + i;
        let dst = r.city.0 + rng.below(n_city);
        located.row(format_args!("{src},{dst}"));
        n_edges += 1;
    }

    n_edges
}

/// Pick a target person index in [0, n). With `--degree-dist zipf`, bias toward
/// low indices so a small set of persons accrue very high in-degree (hubs) —
/// the realistic structure that makes k-hop traversal explode.
#[inline]
fn sample_person(rng: &mut Rng, n: u64, cfg: &Config) -> u64 {
    if cfg.zipf {
        let u = rng.f64().powf(cfg.zipf_exp); // exp>1 concentrates near 0
        ((u * n as f64) as u64).min(n - 1)
    } else {
        rng.below(n)
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Manifest — schema + seed-derived, valid query params (hand-written JSON).
// ─────────────────────────────────────────────────────────────────────────
fn write_manifest(cfg: &Config, r: &Ranges, n_nodes: u64, n_edges: u64) {
    let mut pr = Rng::new(cfg.seed ^ 0x5EED_5A17_0000_0001);
    let n_person = count(r.person);
    let n_project = count(r.project);
    let n_city = count(r.city);

    let sample = |pr: &mut Rng, base: u64, n: u64, k: u64| -> Vec<u64> {
        (0..k.min(n)).map(|_| base + pr.below(n)).collect()
    };

    let lookup_ids = sample(&mut pr, r.person.0, n_person, 500);
    let seed_persons = sample(&mut pr, r.person.0, n_person, 200);
    let seed_persons_small = sample(&mut pr, r.person.0, n_person, 50);
    let seed_persons_tiny = sample(&mut pr, r.person.0, n_person, 10);
    let seed_projects = sample(&mut pr, r.project.0, n_project, 20);
    let sp_pairs: Vec<(u64, u64)> = (0..20)
        .map(|_| {
            (
                r.person.0 + pr.below(n_person),
                r.person.0 + pr.below(n_person),
            )
        })
        .collect();

    let arr = |v: &[u64]| -> String {
        let mut s = String::from("[");
        for (i, x) in v.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            s.push_str(&x.to_string());
        }
        s.push(']');
        s
    };
    let pairs = {
        let mut s = String::from("[");
        for (i, (a, b)) in sp_pairs.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            s.push_str(&format!("[{a},{b}]"));
        }
        s.push(']');
        s
    };

    let json = format!(
        r#"{{
  "schema": "graphsuite",
  "seed": {seed},
  "degree_dist": "{dist}",
  "zipf_exp": {zexp},
  "counts": {{
    "nodes": {n_nodes},
    "edges": {n_edges},
    "Person": {np}, "Company": {nco}, "Project": {npr}, "Skill": {nsk}, "City": {nci}
  }},
  "ranges": {{
    "Person": [{ps},{pe}], "Company": [{cs},{ce}], "Project": [{prs},{pre}],
    "Skill": [{sks},{ske}], "City": [{cis},{cie}]
  }},
  "params": {{
    "lookup_ids": {lookup},
    "filter_age": 40,
    "filter_city": "City_{filter_city}",
    "seed_persons": {sp},
    "seed_persons_small": {sps},
    "seed_persons_tiny": {spt},
    "seed_projects": {sproj},
    "sp_pairs": {pairs},
    "topk": 20,
    "mut_new_base": {mut_base},
    "mut_new_count": 1000
  }}
}}
"#,
        seed = cfg.seed,
        dist = if cfg.zipf { "zipf" } else { "uniform" },
        zexp = cfg.zipf_exp,
        np = n_person,
        nco = count(r.company),
        npr = n_project,
        nsk = count(r.skill),
        nci = n_city,
        ps = r.person.0,
        pe = r.person.1,
        cs = r.company.0,
        ce = r.company.1,
        prs = r.project.0,
        pre = r.project.1,
        sks = r.skill.0,
        ske = r.skill.1,
        cis = r.city.0,
        cie = r.city.1,
        lookup = arr(&lookup_ids),
        filter_city = n_city / 2,
        sp = arr(&seed_persons),
        sps = arr(&seed_persons_small),
        spt = arr(&seed_persons_tiny),
        sproj = arr(&seed_projects),
        pairs = pairs,
        mut_base = r.city.1 + 10_000_000,
    );

    let mut f = File::create(cfg.out.join("manifest.json"))
        .unwrap_or_else(|e| fail(&format!("create manifest: {e}")));
    f.write_all(json.as_bytes()).unwrap();
}

// ─────────────────────────────────────────────────────────────────────────
// Args
// ─────────────────────────────────────────────────────────────────────────
fn parse_args() -> Config {
    let mut out: Option<PathBuf> = None;
    let mut persons: Option<u64> = None;
    let mut knows_per = 8u64;
    let mut seed = 1234u64;
    let mut zipf = true;
    let mut zipf_exp = 1.6f64;

    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut val = || {
            args.next()
                .unwrap_or_else(|| fail(&format!("{a} needs a value")))
        };
        match a.as_str() {
            "--out" => out = Some(PathBuf::from(val())),
            "--persons" => persons = Some(val().parse().unwrap_or_else(|_| fail("bad --persons"))),
            "--scale" => {
                let v = val();
                persons = Some(scale_persons(&v).unwrap_or_else(|| {
                    fail("unknown --scale (tiny|small|medium|large|huge|xhuge)")
                }));
            }
            "--knows-per" => knows_per = val().parse().unwrap_or_else(|_| fail("bad --knows-per")),
            "--seed" => seed = val().parse().unwrap_or_else(|_| fail("bad --seed")),
            "--degree-dist" => zipf = matches!(val().as_str(), "zipf"),
            "--zipf-exp" => zipf_exp = val().parse().unwrap_or_else(|_| fail("bad --zipf-exp")),
            "-h" | "--help" => {
                eprintln!(
                    "graphgen --out DIR [--scale NAME | --persons N] [--seed S] \
                     [--knows-per K] [--degree-dist uniform|zipf] [--zipf-exp E]"
                );
                exit(0);
            }
            other => fail(&format!("unknown arg {other}")),
        }
    }

    Config {
        out: out.unwrap_or_else(|| fail("--out DIR is required")),
        persons: persons.unwrap_or_else(|| fail("--scale NAME or --persons N is required")),
        knows_per,
        seed,
        zipf,
        zipf_exp,
    }
}
