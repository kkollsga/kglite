use super::*;
use serde_json::json;

fn blueprint(ops: serde_json::Value) -> Blueprint {
    serde_json::from_value(json!({"nodes": {}, "compute": ops})).unwrap()
}

#[test]
fn computed_names_are_portable_unique_and_order_independent() {
    let tmp = tempfile::tempdir().unwrap();
    let ops = json!([
        {"op":"derive","from":"A-B","set":{}},
        {"op":"derive","from":"A_B","set":{}},
        {"op":"derive","from":"Name","set":{}},
        {"op":"derive","from":"name","set":{}},
        {"op":"derive","from":"Ordinary","set":{}},
        {"op":"filter","from":"S","into":"X-Y","where":"true"},
        {"op":"filter","from":"S","into":"X_Y","where":"true"},
        {"op":"aggregate","from":"S","into":"Total-A","group_by":[],"agg":{}},
        {"op":"aggregate","from":"S","into":"Total_A","group_by":[],"agg":{}},
        {"op":"chain","from":"S1","edge":"NEXT","group_by":[],"order_by":"id"},
        {"op":"chain","from":"S2","edge":"NEXT","group_by":[],"order_by":"id"},
        {"op":"calendar","type":"Date-A","start":"2026-01-01","end":"2026-01-01"},
        {"op":"calendar","type":"Date_A","start":"2026-01-02","end":"2026-01-02"}
    ]);
    let mut bp = blueprint(ops);
    let a = ComputePaths::new(&bp, tmp.path(), &bp.compute).unwrap();
    bp.compute.reverse();
    let b = ComputePaths::new(&bp, tmp.path(), &bp.compute).unwrap();
    assert_eq!(a.0, b.0);
    assert_eq!(
        a.relative(&Output::Derive("Ordinary".into())),
        "computed/Ordinary_derived.csv"
    );
    let folded: BTreeSet<_> = a.0.values().map(|name| name.to_ascii_lowercase()).collect();
    assert_eq!(folded.len(), a.0.len());
    assert_eq!(a.0.len(), 15);
    assert!(a
        .0
        .values()
        .all(|name| name.starts_with("computed/") && name.len() <= 129));
}

#[test]
fn computed_paths_reserve_existing_entries_and_all_declared_source_kinds() {
    let tmp = tempfile::tempdir().unwrap();
    std::fs::create_dir(tmp.path().join("computed")).unwrap();
    let reserved = tmp.path().join("computed/T_DERIVED.CSV");
    std::fs::write(&reserved, b"unrelated bytes").unwrap();
    let bp: Blueprint = serde_json::from_value(json!({
        "files":{"source":{"path":"computed/compute_0.csv"}},
        "nodes":{"Outer":{"csv":"computed/compute_1.csv","sub_nodes":{"Inner":{"csv":"computed/compute_2.csv"}},
            "connections":{"junction_edges":{"R":{"csv":"computed/compute_3.csv","source_fk":"id","target":"Outer","target_fk":"id"}}}}},
        "compute":[{"op":"derive","from":"T","set":{}}]
    })).unwrap();
    let paths = ComputePaths::new(&bp, tmp.path(), &bp.compute).unwrap();
    assert_eq!(
        paths.relative(&Output::Derive("T".into())),
        "computed/compute_4.csv"
    );
    assert_eq!(std::fs::read(&reserved).unwrap(), b"unrelated bytes");
}

#[test]
fn repeated_owner_reuses_only_current_map_and_fresh_invocation_reserves_old_output() {
    let tmp = tempfile::tempdir().unwrap();
    std::fs::create_dir(tmp.path().join("computed")).unwrap();
    let bp = blueprint(json!([
        {"op":"derive","from":"T","set":{}}, {"op":"derive","from":"T","set":{}}
    ]));
    let first = ComputePaths::new(&bp, tmp.path(), &bp.compute).unwrap();
    assert_eq!(first.0.len(), 1);
    let relative = first.relative(&Output::Derive("T".into()));
    std::fs::write(tmp.path().join(&relative), b"prior completed output").unwrap();
    assert_eq!(first.relative(&Output::Derive("T".into())), relative);
    let next = ComputePaths::new(&bp, tmp.path(), &bp.compute).unwrap();
    assert_ne!(next.relative(&Output::Derive("T".into())), relative);
    assert_eq!(
        std::fs::read(tmp.path().join(relative)).unwrap(),
        b"prior completed output"
    );
}
