//! IMPLEMENTS / EXTENDS / HAS_METHOD edges from TypeRelationships.
//!
//! Also produces synthetic external Trait / Class node records for
//! target types referenced but not defined locally.

use crate::code_tree::models::{ClassInfo, FileInfo, InterfaceInfo, TypeRelationship};
use std::collections::{HashMap, HashSet};

pub struct ImplementsEdge {
    pub type_name: String,
    pub interface_name: String,
}

pub struct ExtendsEdge {
    pub child_name: String,
    pub parent_name: String,
}

pub struct HasMethodEdge {
    pub owner: String,
    pub method: String,
}

pub struct ExternalNode {
    pub qualified_name: String,
    pub name: String,
}

pub struct TypeEdgeOutput {
    pub implements: Vec<ImplementsEdge>,
    pub extends: Vec<ExtendsEdge>,
    pub has_method: Vec<HasMethodEdge>,
    pub external_traits: Vec<ExternalNode>,
    pub external_classes: Vec<ExternalNode>,
}

/// Kind of a known parsed type. Drives kind-aware target resolution: an
/// `implements` relationship prefers an `Interface` candidate when the bare
/// target name matches multiple namespaces; `extends` prefers a class-like
/// candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Class,
    Interface,
}

/// Index over every parsed type: bare name → list of `(qname, kind)` pairs.
/// Built once and reused for every relationship's target lookup.
struct TypeIndex {
    by_name: HashMap<String, Vec<(String, Kind)>>,
    /// `qname → file_path` for every parsed type, used to derive a
    /// relationship's source-file context (and thus its using/import list).
    qname_to_file: HashMap<String, String>,
    /// `file_path → list of namespace prefixes` declared as imports/usings.
    /// Used to score multi-match resolution: candidates whose qname starts
    /// with `<using>.` or with the source's own namespace win.
    file_imports: HashMap<String, Vec<String>>,
    /// `qname → namespace prefix` (everything before the last `.`/`::`).
    /// Used as the implicit own-namespace import for every file.
    qname_to_namespace: HashMap<String, String>,
}

impl TypeIndex {
    fn build(files: &[FileInfo], classes: &[ClassInfo], interfaces: &[InterfaceInfo]) -> Self {
        let mut by_name: HashMap<String, Vec<(String, Kind)>> = HashMap::new();
        let mut qname_to_file: HashMap<String, String> = HashMap::new();
        let mut qname_to_namespace: HashMap<String, String> = HashMap::new();

        for c in classes {
            let kind = Kind::Class;
            by_name
                .entry(c.name.clone())
                .or_default()
                .push((c.qualified_name.clone(), kind));
            qname_to_file.insert(c.qualified_name.clone(), c.file_path.clone());
            qname_to_namespace.insert(c.qualified_name.clone(), namespace_of(&c.qualified_name));
        }
        for i in interfaces {
            let kind = Kind::Interface;
            by_name
                .entry(i.name.clone())
                .or_default()
                .push((i.qualified_name.clone(), kind));
            qname_to_file.insert(i.qualified_name.clone(), i.file_path.clone());
            qname_to_namespace.insert(i.qualified_name.clone(), namespace_of(&i.qualified_name));
        }

        let mut file_imports: HashMap<String, Vec<String>> = HashMap::new();
        for f in files {
            if !f.imports.is_empty() {
                file_imports.insert(f.path.clone(), f.imports.clone());
            }
        }

        TypeIndex {
            by_name,
            qname_to_file,
            file_imports,
            qname_to_namespace,
        }
    }

    /// Resolve a bare or qualified name against the index, scoring
    /// candidates by namespace proximity to the source. Returns
    /// `(resolved_qname, kind_if_known)`. When nothing matches, returns
    /// the input verbatim with `None` kind.
    fn resolve(
        &self,
        name: &str,
        source_qname: &str,
        prefer: Option<Kind>,
    ) -> (String, Option<Kind>) {
        // 1. exact-qname hit (already-qualified).
        if let Some(file) = self.qname_to_file.get(name) {
            // We know this qname; recover its kind from `by_name`.
            let bare = name.rsplit_once('.').map(|(_, n)| n).unwrap_or(name);
            let kind = self
                .by_name
                .get(bare)
                .and_then(|cands| cands.iter().find(|(q, _)| q == name).map(|(_, k)| *k));
            let _ = file;
            return (name.to_string(), kind);
        }

        // 2. bare-name hit.
        let Some(candidates) = self.by_name.get(name) else {
            return (name.to_string(), None);
        };
        if candidates.len() == 1 {
            let (qn, k) = &candidates[0];
            return (qn.clone(), Some(*k));
        }

        // 3. multi-match: score by (kind preference, namespace proximity).
        let source_file = self.qname_to_file.get(source_qname);
        let source_ns = self.qname_to_namespace.get(source_qname).cloned();
        let imports: Vec<String> = source_file
            .and_then(|p| self.file_imports.get(p))
            .cloned()
            .unwrap_or_default();

        // Build the set of prefixes that count as "in scope" for the source:
        // its own namespace, plus every imported namespace.
        let mut in_scope: Vec<String> = imports;
        if let Some(ns) = source_ns {
            if !ns.is_empty() && !in_scope.contains(&ns) {
                in_scope.push(ns);
            }
        }

        let score = |qn: &str, k: Kind| -> i32 {
            let mut s = 0;
            if let Some(p) = prefer {
                if p == k {
                    s += 100;
                }
            }
            // Strongest namespace match: qname starts with `<scope>.` for any
            // scope. Pick the longest matching prefix.
            let mut best_prefix = 0usize;
            for scope in &in_scope {
                if qn.len() > scope.len() + 1
                    && qn.starts_with(scope.as_str())
                    && qn.as_bytes()[scope.len()] == b'.'
                    && scope.len() > best_prefix
                {
                    best_prefix = scope.len();
                }
            }
            s += best_prefix as i32;
            s
        };

        let best = candidates
            .iter()
            .max_by_key(|(qn, k)| score(qn.as_str(), *k))
            .expect("non-empty");
        (best.0.clone(), Some(best.1))
    }
}

fn namespace_of(qname: &str) -> String {
    // C# / Java / Python use `.`; Rust / C++ use `::`. Try `.` first since
    // C# is the dominant case for namespace-aware resolution; fall back to `::`.
    if let Some((ns, _)) = qname.rsplit_once('.') {
        return ns.to_string();
    }
    if let Some((ns, _)) = qname.rsplit_once("::") {
        return ns.to_string();
    }
    String::new()
}

fn resolve_owner(qualified_name: &str, name_to_qname: &HashMap<String, String>) -> String {
    // Separator ordering matters — match the longest specific separator
    // first so `App\Foo::method` (cross-PHP-namespace static call) finds
    // `::` before falling through to `\`. `\` (single backslash) is
    // PHP's namespace separator and was added in 0.9.36 alongside the
    // PHP parser.
    for sep in ["::", ".", "\\"] {
        if let Some(idx) = qualified_name.rfind(sep) {
            let owner = &qualified_name[..idx];
            let simple = owner
                .rfind(sep)
                .map(|i| &owner[i + sep.len()..])
                .unwrap_or(owner);
            return name_to_qname
                .get(simple)
                .cloned()
                .unwrap_or_else(|| owner.to_string());
        }
    }
    qualified_name.to_string()
}

/// Build IMPLEMENTS/EXTENDS/HAS_METHOD edges + external trait/class stubs.
///
/// Resolution is kind-aware: `implements` relationships prefer Interface
/// targets when multiple namespaces define the same bare name; `extends`
/// prefer class-like targets. This eliminates the bulk of the
/// `Class -[IMPLEMENTS]-> Class` mis-typed rows that appeared on
/// dotnet/runtime when an interface name collided with an unrelated class.
///
/// Auto-reroute: if a parser reports `extends X` but `X` resolves to an
/// Interface (e.g. C#'s `class Foo : IDisposable` with no base class —
/// the parser unconditionally calls the first base "extends"), the edge
/// is re-emitted as `implements X` instead.
pub fn build_type_edges(
    relationships: &[TypeRelationship],
    files: &[FileInfo],
    classes: &[ClassInfo],
    interfaces: &[InterfaceInfo],
    name_to_qname: &mut HashMap<String, String>,
) -> TypeEdgeOutput {
    let known_interfaces: HashSet<String> = interfaces.iter().map(|i| i.name.clone()).collect();
    let known_classes: HashSet<String> = classes.iter().map(|c| c.name.clone()).collect();
    let index = TypeIndex::build(files, classes, interfaces);

    let mut implements: Vec<ImplementsEdge> = Vec::new();
    let mut extends: Vec<ExtendsEdge> = Vec::new();
    let mut has_method: Vec<HasMethodEdge> = Vec::new();
    let mut seen_impl: HashSet<(String, String)> = HashSet::new();
    let mut seen_ext: HashSet<(String, String)> = HashSet::new();
    let mut external_traits: HashMap<String, ExternalNode> = HashMap::new();
    let mut external_classes: HashMap<String, ExternalNode> = HashMap::new();

    for tr in relationships {
        match tr.relationship.as_str() {
            "inherent" => {
                for method in &tr.methods {
                    has_method.push(HasMethodEdge {
                        owner: resolve_owner(&method.qualified_name, name_to_qname),
                        method: method.qualified_name.clone(),
                    });
                }
            }
            "implements" => {
                let Some(target) = tr.target_type.as_ref() else {
                    continue;
                };
                let key = (tr.source_type.clone(), target.clone());
                if seen_impl.insert(key) {
                    let (resolved_target, _kind) =
                        index.resolve(target, &tr.source_type, Some(Kind::Interface));
                    let resolved_source = name_to_qname
                        .get(&tr.source_type)
                        .cloned()
                        .unwrap_or_else(|| tr.source_type.clone());
                    implements.push(ImplementsEdge {
                        type_name: resolved_source,
                        interface_name: resolved_target.clone(),
                    });
                    if !known_interfaces.contains(target) {
                        external_traits
                            .entry(resolved_target.clone())
                            .or_insert(ExternalNode {
                                qualified_name: resolved_target.clone(),
                                name: target.clone(),
                            });
                        name_to_qname
                            .entry(target.clone())
                            .or_insert(resolved_target);
                    }
                }
                for method in &tr.methods {
                    has_method.push(HasMethodEdge {
                        owner: resolve_owner(&method.qualified_name, name_to_qname),
                        method: method.qualified_name.clone(),
                    });
                }
            }
            "extends" => {
                let Some(target) = tr.target_type.as_ref() else {
                    continue;
                };
                let (resolved_target, kind) =
                    index.resolve(target, &tr.source_type, Some(Kind::Class));

                // Auto-reroute: parser reported "extends" but target is
                // really an Interface. Common in C# where `class Foo :
                // IBar` is emitted as extends(IBar) when there's no base
                // class — IBar is genuinely implements, not extends.
                if matches!(kind, Some(Kind::Interface)) {
                    let key = (tr.source_type.clone(), target.clone());
                    if seen_impl.insert(key) {
                        let resolved_source = name_to_qname
                            .get(&tr.source_type)
                            .cloned()
                            .unwrap_or_else(|| tr.source_type.clone());
                        implements.push(ImplementsEdge {
                            type_name: resolved_source,
                            interface_name: resolved_target.clone(),
                        });
                        if !known_interfaces.contains(target) {
                            external_traits.entry(resolved_target.clone()).or_insert(
                                ExternalNode {
                                    qualified_name: resolved_target.clone(),
                                    name: target.clone(),
                                },
                            );
                            name_to_qname
                                .entry(target.clone())
                                .or_insert(resolved_target);
                        }
                    }
                    continue;
                }

                let key = (tr.source_type.clone(), target.clone());
                if seen_ext.insert(key) {
                    let resolved_source = name_to_qname
                        .get(&tr.source_type)
                        .cloned()
                        .unwrap_or_else(|| tr.source_type.clone());
                    extends.push(ExtendsEdge {
                        child_name: resolved_source.clone(),
                        parent_name: resolved_target.clone(),
                    });
                    if !known_classes.contains(&tr.source_type) {
                        external_classes
                            .entry(resolved_source.clone())
                            .or_insert(ExternalNode {
                                qualified_name: resolved_source,
                                name: tr.source_type.clone(),
                            });
                    }
                    if !known_classes.contains(target) {
                        external_classes
                            .entry(resolved_target.clone())
                            .or_insert(ExternalNode {
                                qualified_name: resolved_target,
                                name: target.clone(),
                            });
                    }
                }
            }
            _ => {}
        }
    }

    TypeEdgeOutput {
        implements,
        extends,
        has_method,
        external_traits: external_traits.into_values().collect(),
        external_classes: external_classes.into_values().collect(),
    }
}
