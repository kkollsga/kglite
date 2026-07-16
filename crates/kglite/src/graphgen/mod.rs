//! Streaming synthetic property-graph generator — bundled so any binding can
//! produce a realistic benchmark / demo graph in one call (mirrors how
//! code-graph building lives in the external codingest project).
//!
//! Emits the canonical org/social schema (Person/Company/Project/Skill/City +
//! KNOWS/WORKS_AT/CONTRIBUTES_TO/HAS_SKILL/OWNS/DEPENDS_ON/LOCATED_IN) as one
//! CSV per type plus a `manifest.json` (schema, counts, seed-derived query
//! params). Every engine that loads the same bytes gets the same graph, so
//! cross-engine result-parity holds by construction.
//!
//! **Bounded memory at any scale:** nodes and edges are streamed row-by-row to
//! disk; the only resident state is the RNG, small per-source dedup sets, and
//! counters — a 50M-node graph generates in the same RAM as a 1k one. The
//! generation is deterministic per `seed`.
//!
//! This is the library form of the former standalone `benchmarks/graphgen`
//! crate; the generator logic is unchanged.

use std::collections::HashSet;
use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};

// ─── Deterministic PRNG — splitmix64 (no deps, reproducible per seed) ────────
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
    #[inline]
    fn below(&mut self, n: u64) -> u64 {
        if n == 0 {
            0
        } else {
            self.next_u64() % n
        }
    }
    #[inline]
    fn f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    #[inline]
    fn range_i64(&mut self, lo: i64, hi: i64) -> i64 {
        lo + self.below((hi - lo + 1) as u64) as i64
    }
    #[inline]
    fn pick<'a, T>(&mut self, xs: &'a [T]) -> &'a T {
        &xs[self.below(xs.len() as u64) as usize]
    }
}

// ─── Schema — categorical value pools ────────────────────────────────────────
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

/// Embedding-vector width on Person nodes (the `embedding` CSV column). Small
/// enough to keep the CSV bounded at scale, wide enough for a meaningful
/// vector-kNN benchmark. Reported in the manifest as `embedding_dim`.
const EMB_DIM: usize = 16;

/// One node type's CSV layout — the load plan a binding needs to ingest it.
pub struct NodeType {
    pub name: &'static str,
    pub csv: &'static str,
    pub id_column: &'static str,
    pub title_column: &'static str,
}

/// One edge type's CSV layout (always `src,dst` of node ids).
pub struct EdgeType {
    pub name: &'static str,
    pub csv: &'static str,
    pub src_type: &'static str,
    pub dst_type: &'static str,
}

/// The fixed schema emitted by the generator — node/edge types in load order.
/// Surfaced so a loader (the Python `kglite.graphgen`, a binding, a test) can
/// ingest the CSVs without hard-coding the layout.
pub const NODE_TYPES: &[NodeType] = &[
    NodeType {
        name: "City",
        csv: "City.csv",
        id_column: "gid",
        title_column: "name",
    },
    NodeType {
        name: "Skill",
        csv: "Skill.csv",
        id_column: "gid",
        title_column: "name",
    },
    NodeType {
        name: "Company",
        csv: "Company.csv",
        id_column: "gid",
        title_column: "name",
    },
    NodeType {
        name: "Project",
        csv: "Project.csv",
        id_column: "gid",
        title_column: "name",
    },
    NodeType {
        name: "Person",
        csv: "Person.csv",
        id_column: "gid",
        title_column: "name",
    },
];
pub const EDGE_TYPES: &[EdgeType] = &[
    EdgeType {
        name: "KNOWS",
        csv: "KNOWS.csv",
        src_type: "Person",
        dst_type: "Person",
    },
    EdgeType {
        name: "WORKS_AT",
        csv: "WORKS_AT.csv",
        src_type: "Person",
        dst_type: "Company",
    },
    EdgeType {
        name: "CONTRIBUTES_TO",
        csv: "CONTRIBUTES_TO.csv",
        src_type: "Person",
        dst_type: "Project",
    },
    EdgeType {
        name: "HAS_SKILL",
        csv: "HAS_SKILL.csv",
        src_type: "Person",
        dst_type: "Skill",
    },
    EdgeType {
        name: "OWNS",
        csv: "OWNS.csv",
        src_type: "Company",
        dst_type: "Project",
    },
    EdgeType {
        name: "DEPENDS_ON",
        csv: "DEPENDS_ON.csv",
        src_type: "Project",
        dst_type: "Project",
    },
    EdgeType {
        name: "LOCATED_IN",
        csv: "LOCATED_IN.csv",
        src_type: "Company",
        dst_type: "City",
    },
];

// ─── Config ──────────────────────────────────────────────────────────────────
/// Generator parameters. Build from a named scale via [`GraphGenConfig::from_scale`]
/// or set `persons` directly.
#[derive(Clone, Debug)]
pub struct GraphGenConfig {
    /// Number of Person nodes — drives every other type's size.
    pub persons: u64,
    /// Average KNOWS out-degree per person.
    pub knows_per: u64,
    /// Deterministic seed.
    pub seed: u64,
    /// Zipf degree distribution (high-degree hubs) vs uniform.
    pub zipf: bool,
    /// Zipf skew exponent (>1 → stronger hubs).
    pub zipf_exp: f64,
}

impl Default for GraphGenConfig {
    fn default() -> Self {
        GraphGenConfig {
            persons: 20_000,
            knows_per: 8,
            seed: 1234,
            zipf: true,
            zipf_exp: 1.6,
        }
    }
}

impl GraphGenConfig {
    /// Person count for a named scale, or `None` if unknown.
    pub fn scale_persons(name: &str) -> Option<u64> {
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

    /// Config for a named scale (`tiny`/`small`/`medium`/`large`/`huge`/`xhuge`),
    /// other fields at their defaults.
    pub fn from_scale(name: &str) -> Option<Self> {
        Self::scale_persons(name).map(|persons| GraphGenConfig {
            persons,
            ..Default::default()
        })
    }
}

/// What a generation run produced.
#[derive(Clone, Debug)]
pub struct GraphGenStats {
    pub nodes: u64,
    pub edges: u64,
    pub out_dir: PathBuf,
}

// Contiguous global-id ranges, allocated in one pass.
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
    fn create(dir: &Path, name: &str, header: &str) -> io::Result<Csv> {
        let f = File::create(dir.join(format!("{name}.csv")))?;
        let mut w = BufWriter::with_capacity(1 << 20, f);
        writeln!(w, "{header}")?;
        Ok(Csv(w))
    }
    #[inline]
    fn row(&mut self, args: std::fmt::Arguments) -> io::Result<()> {
        self.0.write_fmt(args)?;
        self.0.write_all(b"\n")
    }
}

/// Generate the synthetic graph, streaming CSVs + `manifest.json` into `out_dir`
/// (created if needed). Bounded memory regardless of scale.
pub fn generate_to_dir(cfg: &GraphGenConfig, out_dir: &Path) -> io::Result<GraphGenStats> {
    fs::create_dir_all(out_dir)?;
    let r = Ranges::alloc(cfg.persons);
    let n_edges = generate(cfg, &r, out_dir)?;
    let n_nodes =
        count(r.person) + count(r.company) + count(r.project) + count(r.skill) + count(r.city);
    write_manifest(cfg, &r, n_nodes, n_edges, out_dir)?;
    Ok(GraphGenStats {
        nodes: n_nodes,
        edges: n_edges,
        out_dir: out_dir.to_path_buf(),
    })
}

fn generate(cfg: &GraphGenConfig, r: &Ranges, dir: &Path) -> io::Result<u64> {
    let n_city = count(r.city);
    let n_company = count(r.company);
    let n_project = count(r.project);
    let n_skill = count(r.skill);
    let n_person = count(r.person);

    let mut rng = Rng::new(cfg.seed);

    // ---- nodes ----
    // City carries geometry (latitude/longitude) for the geospatial workloads.
    let mut city = Csv::create(dir, "City", "gid,name,population,region,latitude,longitude")?;
    for i in 0..n_city {
        let gid = r.city.0 + i;
        let pop = rng.range_i64(5_000, 9_000_000);
        let region = rng.pick(REGIONS);
        let lat = rng.f64() * 130.0 - 60.0; // -60 .. 70
        let lon = rng.f64() * 340.0 - 170.0; // -170 .. 170
        city.row(format_args!(
            "{gid},City_{i},{pop},{region},{lat:.5},{lon:.5}"
        ))?;
    }

    let mut skill = Csv::create(dir, "Skill", "gid,name,category")?;
    for i in 0..n_skill {
        let gid = r.skill.0 + i;
        skill.row(format_args!(
            "{gid},Skill_{i},{}",
            rng.pick(SKILL_CATEGORIES)
        ))?;
    }

    let mut company = Csv::create(dir, "Company", "gid,name,industry,size")?;
    for i in 0..n_company {
        let gid = r.company.0 + i;
        company.row(format_args!(
            "{gid},Company_{i},{},{}",
            rng.pick(INDUSTRIES),
            rng.range_i64(5, 50_000)
        ))?;
    }

    let mut project = Csv::create(dir, "Project", "gid,name,budget,status")?;
    for i in 0..n_project {
        let gid = r.project.0 + i;
        project.row(format_args!(
            "{gid},Project_{i},{:.2},{}",
            rng.f64() * 1_000_000.0,
            rng.pick(PROJECT_STATUS)
        ))?;
    }

    // Person carries an embedding vector (quoted JSON array) for vector-kNN.
    let mut person = Csv::create(
        dir,
        "Person",
        "gid,name,age,city,joined_year,active,score,embedding",
    )?;
    for i in 0..n_person {
        let gid = r.person.0 + i;
        let age = rng.range_i64(18, 80);
        let city_idx = rng.below(n_city);
        let joined = rng.range_i64(2000, 2025);
        let active = rng.below(2);
        let score = rng.f64() * 100.0;
        let mut emb = String::from("\"[");
        for d in 0..EMB_DIM {
            if d > 0 {
                emb.push(',');
            }
            emb.push_str(&format!("{:.4}", rng.f64() * 2.0 - 1.0));
        }
        emb.push_str("]\"");
        person.row(format_args!(
            "{gid},Person_{i},{age},City_{city_idx},{joined},{active},{score:.4},{emb}",
        ))?;
    }

    // ---- edges (streamed; bounded per-source dedup) ----
    let mut n_edges = 0u64;

    let mut knows = Csv::create(dir, "KNOWS", "src,dst")?;
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
            knows.row(format_args!("{src},{dst}"))?;
            made += 1;
            n_edges += 1;
        }
    }

    let mut works = Csv::create(dir, "WORKS_AT", "src,dst")?;
    for i in 0..n_person {
        let src = r.person.0 + i;
        let dst = r.company.0 + rng.below(n_company);
        works.row(format_args!("{src},{dst}"))?;
        n_edges += 1;
    }

    let mut contrib = Csv::create(dir, "CONTRIBUTES_TO", "src,dst")?;
    for i in 0..n_person {
        let src = r.person.0 + i;
        let k = 1 + rng.below(2);
        for _ in 0..k {
            let dst = r.project.0 + rng.below(n_project);
            contrib.row(format_args!("{src},{dst}"))?;
            n_edges += 1;
        }
    }

    let mut hasskill = Csv::create(dir, "HAS_SKILL", "src,dst")?;
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
            hasskill.row(format_args!("{src},{dst}"))?;
            made += 1;
            n_edges += 1;
        }
    }

    let mut owns = Csv::create(dir, "OWNS", "src,dst")?;
    let owns_per = (n_project / n_company).max(1);
    for i in 0..n_company {
        let src = r.company.0 + i;
        for _ in 0..owns_per {
            let dst = r.project.0 + rng.below(n_project);
            owns.row(format_args!("{src},{dst}"))?;
            n_edges += 1;
        }
    }

    let mut depends = Csv::create(dir, "DEPENDS_ON", "src,dst")?;
    for i in 1..n_project {
        let src = r.project.0 + i;
        let k = rng.below(4);
        for _ in 0..k {
            let dst = r.project.0 + rng.below(i);
            depends.row(format_args!("{src},{dst}"))?;
            n_edges += 1;
        }
    }

    let mut located = Csv::create(dir, "LOCATED_IN", "src,dst")?;
    for i in 0..n_company {
        let src = r.company.0 + i;
        let dst = r.city.0 + rng.below(n_city);
        located.row(format_args!("{src},{dst}"))?;
        n_edges += 1;
    }

    Ok(n_edges)
}

/// Target person index in `[0, n)`. With zipf, bias toward low indices so a
/// small set of persons accrue very high in-degree (hubs) — the structure that
/// makes k-hop traversal interesting.
#[inline]
fn sample_person(rng: &mut Rng, n: u64, cfg: &GraphGenConfig) -> u64 {
    if cfg.zipf {
        let u = rng.f64().powf(cfg.zipf_exp);
        ((u * n as f64) as u64).min(n - 1)
    } else {
        rng.below(n)
    }
}

fn write_manifest(
    cfg: &GraphGenConfig,
    r: &Ranges,
    n_nodes: u64,
    n_edges: u64,
    dir: &Path,
) -> io::Result<()> {
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
  "embedding_dim": {emb_dim},
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
        emb_dim = EMB_DIM,
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

    let mut f = File::create(dir.join("manifest.json"))?;
    f.write_all(json.as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generates_deterministic_schema() {
        let tmp = std::env::temp_dir().join(format!("kglite_graphgen_test_{}", std::process::id()));
        let cfg = GraphGenConfig::from_scale("tiny").unwrap();
        let stats = generate_to_dir(&cfg, &tmp).unwrap();
        assert!(stats.nodes > 1_000);
        assert!(stats.edges > 1_000);
        // every declared CSV + the manifest exists
        for nt in NODE_TYPES {
            assert!(tmp.join(nt.csv).exists(), "missing {}", nt.csv);
        }
        for et in EDGE_TYPES {
            assert!(tmp.join(et.csv).exists(), "missing {}", et.csv);
        }
        assert!(tmp.join("manifest.json").exists());

        // determinism: same seed → byte-identical Person.csv
        let tmp2 = tmp.with_extension("b");
        generate_to_dir(&cfg, &tmp2).unwrap();
        let a = fs::read(tmp.join("Person.csv")).unwrap();
        let b = fs::read(tmp2.join("Person.csv")).unwrap();
        assert_eq!(a, b, "same seed must produce identical bytes");

        let _ = fs::remove_dir_all(&tmp);
        let _ = fs::remove_dir_all(&tmp2);
    }

    #[test]
    fn unknown_scale_is_none() {
        assert!(GraphGenConfig::from_scale("nope").is_none());
        assert_eq!(
            GraphGenConfig::from_scale("medium").unwrap().persons,
            20_000
        );
    }
}
