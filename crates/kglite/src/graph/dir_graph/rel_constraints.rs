//! Declared constraints on *relationships* — the connection-type counterpart of
//! [`super::constraints`].
//!
//! Two kinds are served: presence (`REQUIRE r.p IS NOT NULL`) and property type
//! (`REQUIRE r.p IS :: T`). Uniqueness and RELATIONSHIP KEY are deliberately
//! absent — the engine has no settled multi-edge answer (the bulk loader
//! deduplicates `(type, src, tgt)` while Cypher `CREATE` freely makes parallel
//! edges), so a uniqueness declaration would mean different things on different
//! write paths. The DDL layer refuses them by name.
//!
//! **Its own module, not more of `constraints.rs`.** The stores, the scan and
//! the write-path gates are a separate surface reading separate state, and the
//! node file is near its size ceiling; keeping them apart also keeps the node
//! fast-outs (`has_property_type_constraints` and friends) reading exactly the
//! node stores, so a graph that constrains only relationships pays nothing on
//! the node write path, and vice versa.
//!
//! **Enforcement on new writes arrives with the write-path gates.** What lives
//! here is declaration only: install (validated against the existing data),
//! drop, and list. The per-type fast-out accessors a row gate needs land with
//! that gate — added here first, they would be surface with no caller.

use crate::datatypes::values::Value;
use crate::graph::algorithms::Interrupt;
use crate::graph::constraints::{ConstraintKind, ConstraintViolation, EntityKind};
use crate::graph::property_types::{self, DeclaredType};
use crate::graph::storage::interner::InternedKey;

use super::DirGraph;

/// Relationships visited between two interrupt polls in the declaration scan.
/// The same cadence the Cypher executor's sequential loops and
/// [`crate::graph::parallel::PARALLEL_POLL_INTERVAL`] use; each of those
/// declares its own because the constants sit in modules that cannot see one
/// another. Must stay a power of two — the gate is a mask.
const SCAN_POLL_INTERVAL: usize = 4096;

/// Why a relationship declaration did not install.
///
/// Two outcomes that must not be conflated: the data *disagrees* with the
/// constraint, or the scan never finished. Reporting an interruption as a
/// violation would name offending relationships that were never counted, and
/// installing on an interrupted scan would claim a verification that did not
/// happen — so both refuse, in their own words.
#[derive(Debug)]
pub(crate) enum RelDeclarationError {
    /// Existing relationships violate the requested constraint.
    Violated(Box<ConstraintViolation>),
    /// The query deadline passed, or a binding flipped the cancel flag, before
    /// the scan finished.
    Interrupted(String),
}

impl From<Box<ConstraintViolation>> for RelDeclarationError {
    fn from(violation: Box<ConstraintViolation>) -> Self {
        RelDeclarationError::Violated(violation)
    }
}

pub(crate) type RelDeclarationResult<T> = Result<T, RelDeclarationError>;

impl DirGraph {
    // ========================================================================
    // Presence — `REQUIRE r.p IS NOT NULL`
    // ========================================================================

    /// Whether `property` is declared NOT NULL on `rel_type`.
    pub(crate) fn has_rel_not_null_constraint(&self, rel_type: &str, property: &str) -> bool {
        self.rel_ddl_not_null_constraints
            .contains(&(rel_type.to_string(), property.to_string()))
    }

    /// Every declared relationship presence constraint, `(rel_type, property)`
    /// in deterministic order.
    pub(crate) fn list_rel_not_null_constraints(&self) -> Vec<(String, String)> {
        self.rel_ddl_not_null_constraints.iter().cloned().collect()
    }

    /// Declare `property` present and non-null on every relationship of
    /// `rel_type`. Returns how many relationships were checked.
    ///
    /// Refused, installing nothing, when the existing data already violates it
    /// — the same posture the node declaration takes, because a constraint that
    /// exempts the rows already present is worse than a rejected declaration.
    ///
    /// Idempotent: re-declaring re-verifies and changes nothing, so
    /// `IF NOT EXISTS` and a reload both work.
    pub(crate) fn create_rel_not_null_constraint(
        &mut self,
        rel_type: &str,
        property: &str,
        interrupt: &Interrupt,
    ) -> RelDeclarationResult<usize> {
        let (checked, missing) = self.count_rel_missing_property(rel_type, property, interrupt)?;
        if missing > 0 {
            return Err(RelDeclarationError::Violated(Box::new(
                ConstraintViolation::preexisting_missing(
                    ConstraintKind::NotNull,
                    rel_type,
                    property,
                    missing,
                )
                .on_entity(EntityKind::Relationship),
            )));
        }
        self.rel_ddl_not_null_constraints
            .insert((rel_type.to_string(), property.to_string()));
        Ok(checked)
    }

    /// Withdraw a relationship presence declaration. Reports whether one went.
    pub(crate) fn drop_rel_not_null_constraint(&mut self, rel_type: &str, property: &str) -> bool {
        self.rel_ddl_not_null_constraints
            .remove(&(rel_type.to_string(), property.to_string()))
    }

    // ========================================================================
    // Property type — `REQUIRE r.p IS :: T`
    // ========================================================================

    /// The type declared for `rel_type.property`, if one is.
    pub(crate) fn rel_property_type_for(
        &self,
        rel_type: &str,
        property: &str,
    ) -> Option<DeclaredType> {
        self.rel_ddl_property_type_constraints
            .get(rel_type)?
            .get(property)
            .copied()
    }

    /// Every declared relationship property type as
    /// `(rel_type, property, type)`, in deterministic order.
    pub(crate) fn list_rel_property_type_constraints(&self) -> Vec<(String, String, DeclaredType)> {
        self.rel_ddl_property_type_constraints
            .iter()
            .flat_map(|(rel_type, declared)| {
                declared
                    .iter()
                    .map(move |(property, kind)| (rel_type.clone(), property.clone(), *kind))
            })
            .collect()
    }

    /// Declare `property` on `rel_type` to hold only `declared` values.
    /// Returns how many relationships were checked.
    ///
    /// Refused, installing nothing, when existing relationships hold a value of
    /// another type — mirroring [`Self::create_rel_not_null_constraint`].
    pub(crate) fn create_rel_property_type_constraint(
        &mut self,
        rel_type: &str,
        property: &str,
        declared: DeclaredType,
        interrupt: &Interrupt,
    ) -> RelDeclarationResult<usize> {
        let (checked, violations, sample) =
            self.count_rel_type_violations(rel_type, property, declared, interrupt)?;
        if violations > 0 {
            return Err(RelDeclarationError::Violated(Box::new(
                ConstraintViolation::preexisting_type_mismatch(
                    rel_type,
                    property,
                    declared.name(),
                    sample.unwrap_or("a value of another type"),
                    violations,
                )
                .on_entity(EntityKind::Relationship),
            )));
        }
        self.rel_ddl_property_type_constraints
            .entry(rel_type.to_string())
            .or_default()
            .insert(property.to_string(), declared);
        Ok(checked)
    }

    /// Withdraw a relationship property-type declaration. Reports whether one
    /// went. Removes the type's entry with its last declaration, so the
    /// write-path fast-out returns to `false` once every constraint is dropped.
    pub(crate) fn drop_rel_property_type_constraint(
        &mut self,
        rel_type: &str,
        property: &str,
    ) -> bool {
        let Some(declared) = self.rel_ddl_property_type_constraints.get_mut(rel_type) else {
            return false;
        };
        let removed = declared.remove(property).is_some();
        if declared.is_empty() {
            self.rel_ddl_property_type_constraints.remove(rel_type);
        }
        removed
    }

    // ========================================================================
    // The existing-data scan
    // ========================================================================

    /// Walk every relationship of `rel_type`, handing each one's property slice
    /// to `visit`. Returns how many were visited.
    ///
    /// **Why `for_each_edge_of_conn_type` and not `edge_endpoint_keys`.** The
    /// endpoint-key iterator is the right tool for *counting* by type — it is
    /// what `get_edge_type_counts` uses — but it yields
    /// `(source, target, connection_type)` and no edge index, so there is no
    /// route from one of its items to that edge's properties. This one filters
    /// by type and hands over the property slice in the same pass, and it is
    /// the arena-safe reader on the disk backend: it reads `edge_endpoints` +
    /// `edge_properties` directly instead of materialising a `Box<EdgeData>`
    /// per edge into the per-query arena, which is the same rule
    /// `owned_node_data` follows on the node side. On disk it is also
    /// O(matching edges) rather than O(all edges), through the persisted
    /// `conn_type_index_*` inverted index.
    ///
    /// Interrupt-checked every [`SCAN_POLL_INTERVAL`] relationships: a
    /// declaration on a large graph is an O(E) read, and the node-side scans'
    /// lack of one is a known gap this does not inherit.
    fn for_each_rel_of_type<F>(
        &self,
        rel_type: &str,
        interrupt: &Interrupt,
        mut visit: F,
    ) -> Result<usize, String>
    where
        F: FnMut(&[(InternedKey, Value)]),
    {
        // A type with no edges cannot violate anything. Short-circuited only
        // from a *warm* cache: `get_edge_type_counts` builds an O(E) map when
        // cold, so consulting it unconditionally would pay a whole-graph sweep
        // to avoid one that is often smaller.
        if self.has_edge_type_counts_cache()
            && self
                .get_edge_type_counts()
                .get(rel_type)
                .is_none_or(|count| *count == 0)
        {
            return Ok(0);
        }
        let conn_key = InternedKey::from_str(rel_type);
        let mut visited = 0usize;
        let mut interrupted = false;
        self.graph.for_each_edge_of_conn_type(
            conn_key,
            |_source, _target, _edge_idx, properties| {
                if visited & (SCAN_POLL_INTERVAL - 1) == 0 && interrupt.exceeded() {
                    interrupted = true;
                    return false;
                }
                visited += 1;
                visit(properties);
                true
            },
        );
        if interrupted {
            return Err(format!(
                "declaring a constraint on relationship type '{rel_type}' was interrupted after \
                 {visited} relationships: the declaration is verified against the existing data, \
                 which is a scan of every relationship of the type. Nothing was installed. Raise \
                 the timeout, or declare the constraint before loading the data."
            ));
        }
        Ok(visited)
    }

    /// The value stored for `property` on one relationship, or `None` when it
    /// carries no such property. A stored null reads as `Some(Value::Null)` and
    /// is the caller's to interpret — absent and null are the same thing to a
    /// presence constraint and both fine to a type constraint, and those two
    /// rules disagree about which, so this reports the fact rather than
    /// deciding.
    #[inline]
    fn rel_property(properties: &[(InternedKey, Value)], key: InternedKey) -> Option<&Value> {
        properties
            .iter()
            .find(|(stored, _)| *stored == key)
            .map(|(_, value)| value)
    }

    /// `(relationships_checked, relationships_missing_the_property)`.
    fn count_rel_missing_property(
        &self,
        rel_type: &str,
        property: &str,
        interrupt: &Interrupt,
    ) -> RelDeclarationResult<(usize, usize)> {
        let key = InternedKey::from_str(property);
        let mut missing = 0usize;
        let checked = self
            .for_each_rel_of_type(rel_type, interrupt, |properties| {
                match Self::rel_property(properties, key) {
                    Some(Value::Null) | None => missing += 1,
                    Some(_) => {}
                }
            })
            .map_err(RelDeclarationError::Interrupted)?;
        Ok((checked, missing))
    }

    /// `(checked, violating, one_offending_type_name)` for a candidate
    /// property-type declaration. Absent and null values pass, exactly as they
    /// do on the node side: a type constraint is not an existence constraint.
    fn count_rel_type_violations(
        &self,
        rel_type: &str,
        property: &str,
        declared: DeclaredType,
        interrupt: &Interrupt,
    ) -> RelDeclarationResult<(usize, usize, Option<&'static str>)> {
        let key = InternedKey::from_str(property);
        let mut violations = 0usize;
        let mut sample: Option<&'static str> = None;
        let checked = self
            .for_each_rel_of_type(rel_type, interrupt, |properties| {
                let Some(value) = Self::rel_property(properties, key) else {
                    return;
                };
                if matches!(value, Value::Null) || declared.accepts(value) {
                    return;
                }
                violations += 1;
                sample.get_or_insert_with(|| property_types::value_type_name(value));
            })
            .map_err(RelDeclarationError::Interrupted)?;
        Ok((checked, violations, sample))
    }
}
