//! Relationship uniqueness — `CREATE CONSTRAINT … FOR ()-[r:T]-() REQUIRE r.p IS
//! UNIQUE`, the edge twin of the node `UNIQUE` constraint ([`super::owned_index`]).
//!
//! **Meaning.** No two relationships of type `T` hold the same non-null value of
//! `p`. That is Neo4j's rule and it is well defined on every write path; it says
//! nothing about when two relationships are "the same one", which is why the
//! earlier refusal (the bulk loader deduplicates `(type, source, target)` while
//! `CREATE` makes parallel edges) does not apply: parallel edges are legal as long
//! as their `p` differ.
//!
//! **No structure of its own.** The constraint owns an entry of
//! `rel_property_indices` ([`super::rel_index`]), so the index forks in O(1),
//! the undo journal reverses it, and `MATCH ()-[r:T {p: $v}]->()` reads it.
//! What the constraint adds is the check ([`DirGraph::rel_unique_occupant`]) and
//! ownership, with the same three index states the node side has: user-only,
//! owned-only (hidden from the persisted index list, refuses `DROP INDEX`, goes
//! with `DROP CONSTRAINT`) and adopted (a user declared the index first; it is
//! persisted with the user lists and survives `DROP CONSTRAINT`).
//!
//! **Exact equality.** The probe is the bucket for the claimed `Value`, so `1`
//! and `1.0` are two values and a numeric string is a third, exactly as for the
//! node constraint. A bucket member only counts when it is live and its stored
//! value still equals the probe, so a stale entry can never refuse a legal write.
//!
//! **Scope.** One property, memory and mapped graphs, and no declared temporal
//! relationship type (several versions of "the same" relationship legitimately
//! share a business key there). Each is refused with its reason at declaration.
//! `NULL` and absent values claim nothing.

use petgraph::graph::EdgeIndex;

use super::rel_index::slot_edge;
use super::DirGraph;
use crate::datatypes::values::Value;
use crate::graph::algorithms::Interrupt;
use crate::graph::constraints::{descriptor, ConstraintKind, ConstraintViolation, EntityKind};
use crate::graph::dir_graph::rel_constraints::{RelDeclarationError, RelDeclarationResult};
use crate::graph::schema::IndexKey;
use crate::graph::storage::GraphRead;

impl DirGraph {
    /// The properties `rel_type` declares unique. A range over the ordered set,
    /// so a type that declares nothing costs one lookup and no allocation.
    pub(crate) fn rel_unique_properties<'a>(
        &'a self,
        rel_type: &'a str,
    ) -> impl Iterator<Item = &'a str> + 'a {
        self.rel
            .unique
            .range((rel_type.to_string(), String::new())..)
            .take_while(move |(declared, _)| declared == rel_type)
            .map(|(_, property)| property.as_str())
    }

    /// Whether `rel_type` declares any uniqueness constraint.
    #[inline]
    pub(crate) fn type_has_rel_unique(&self, rel_type: &str) -> bool {
        self.rel_unique_properties(rel_type).next().is_some()
    }

    /// Whether `property` is declared unique on `rel_type`.
    pub(crate) fn has_rel_unique_constraint(&self, rel_type: &str, property: &str) -> bool {
        self.rel
            .unique
            .contains(&(rel_type.to_string(), property.to_string()))
    }

    /// Every declared relationship uniqueness constraint as
    /// `(rel_type, property)`, in deterministic order.
    pub(crate) fn list_rel_unique_constraints(&self) -> Vec<(String, String)> {
        self.rel.unique.iter().cloned().collect()
    }

    /// Declare `property` unique on `rel_type`, building the index it owns (or
    /// adopting a declared one) and refusing when the stored data already holds
    /// a repeated value. Returns the number of relationships indexed.
    ///
    /// Refused, installing nothing, on a disk graph, on a type with a declared
    /// temporal interval, and when the data violates it. Idempotent.
    pub(crate) fn create_rel_unique_constraint(
        &mut self,
        rel_type: &str,
        property: &str,
        interrupt: &Interrupt,
    ) -> RelDeclarationResult<usize> {
        self.refuse_rel_index_on_disk("a UNIQUE constraint")
            .map_err(RelDeclarationError::Refused)?;
        if !crate::graph::features::temporal::edge_configs(self, rel_type).is_empty() {
            return Err(RelDeclarationError::Refused(format!(
                "a UNIQUE constraint on relationship type '{rel_type}' is not supported: the \
                 type declares a validity interval, where several versions of one relationship \
                 legitimately share a value. Declare uniqueness on a type without an interval."
            )));
        }
        self.materialize_indexes();
        let key: IndexKey = (rel_type.to_string(), property.to_string());
        let was_built = self.rel.indices.contains_key(&key);
        if !was_built {
            self.build_rel_property_index(rel_type, property);
        }
        if let Some(violation) = self.scan_rel_unique_violation(&key, interrupt)? {
            if !was_built {
                self.rel.indices.remove(&key);
            }
            return Err(RelDeclarationError::Violated(Box::new(violation)));
        }
        if was_built {
            self.rel.adopted.insert(key.clone());
        }
        self.rel.unique.insert(key.clone());
        Ok(self.rel.indices.get(&key).map_or(0, |index| {
            index.iter().map(|(_, members)| members.len()).sum()
        }))
    }

    /// The first repeated value in the index `key`, as a declaration refusal.
    /// Counts every repeated value so the message can say how many.
    fn scan_rel_unique_violation(
        &self,
        key: &IndexKey,
        interrupt: &Interrupt,
    ) -> RelDeclarationResult<Option<ConstraintViolation>> {
        let Some(index) = self.rel.indices.get(key) else {
            return Ok(None);
        };
        let mut repeated = 0usize;
        let mut sample: Option<Value> = None;
        for (visited, (value, members)) in index.iter().enumerate() {
            if visited % 4096 == 0 && interrupt.exceeded() {
                return Err(RelDeclarationError::Interrupted(format!(
                    "declaring a UNIQUE constraint on {}.{} was interrupted while checking \
                         the existing values. Nothing was installed. Raise the timeout, or \
                         declare the constraint before loading the data.",
                    key.0, key.1
                )));
            }
            if members.len() > 1 {
                repeated += 1;
                sample.get_or_insert_with(|| value.clone());
            }
        }
        Ok(sample.map(|sample| {
            ConstraintViolation::preexisting(
                ConstraintKind::Unique,
                key.0.clone(),
                vec![key.1.clone()],
                repeated,
                vec![sample],
            )
            .on_entity(EntityKind::Relationship)
        }))
    }

    /// Withdraw a relationship uniqueness declaration, and the index it owned
    /// unless a user declared that index too. Reports whether one went.
    pub(crate) fn drop_rel_unique_constraint(&mut self, rel_type: &str, property: &str) -> bool {
        self.materialize_indexes();
        let key: IndexKey = (rel_type.to_string(), property.to_string());
        if !self.rel.unique.remove(&key) {
            return false;
        }
        if !self.rel.adopted.remove(&key) {
            self.rel.indices.remove(&key);
        }
        true
    }

    /// The live edge currently holding `value` under the unique `(rel_type,
    /// property)`, other than `except`, if any.
    pub(crate) fn rel_unique_occupant(
        &self,
        rel_type: &str,
        property: &str,
        value: &Value,
        except: Option<EdgeIndex>,
    ) -> Option<EdgeIndex> {
        let key = (rel_type.to_string(), property.to_string());
        let members = self.rel.indices.get(&key)?.get(value)?;
        members
            .iter()
            .map(|member| slot_edge(*member))
            .find(|edge| {
                Some(*edge) != except
                    && self.edge_is_of_type(*edge, rel_type)
                    && self.read_edge_value(*edge, property).as_ref() == Some(value)
            })
    }

    fn edge_is_of_type(&self, edge: EdgeIndex, rel_type: &str) -> bool {
        self.graph
            .edge_weight(edge)
            .and_then(|data| self.interner.try_resolve(data.connection_type))
            == Some(rel_type)
    }

    /// Gate a relationship about to be created: its presence and type rules,
    /// then its uniqueness.
    pub(crate) fn check_new_rel_row<F>(&mut self, rel_type: &str, read: F) -> Result<(), String>
    where
        F: Fn(&str) -> Option<Value>,
    {
        self.check_rel_row_types(rel_type, &read)?;
        self.check_rel_unique_row(rel_type, read, None)
    }

    /// Gate one relationship's worth of properties against every uniqueness
    /// constraint of its type. `read` returns the value the write leaves for a
    /// property (`None` absent); `except` is the edge being rewritten, which
    /// does not conflict with itself.
    ///
    /// The violation is parked before it is returned so a binding raises a typed
    /// error, as [`Self::check_rel_row`] does.
    pub(crate) fn check_rel_unique_row<F>(
        &mut self,
        rel_type: &str,
        read: F,
        except: Option<EdgeIndex>,
    ) -> Result<(), String>
    where
        F: Fn(&str) -> Option<Value>,
    {
        let properties: Vec<String> = self
            .rel_unique_properties(rel_type)
            .map(str::to_string)
            .collect();
        for property in properties {
            let Some(value) = read(&property).filter(|v| !matches!(v, Value::Null)) else {
                continue;
            };
            if self
                .rel_unique_occupant(rel_type, &property, &value, except)
                .is_some()
            {
                let violation = ConstraintViolation::duplicate(
                    ConstraintKind::Unique,
                    rel_type,
                    vec![property],
                    vec![value],
                )
                .on_entity(EntityKind::Relationship);
                return Err(self.record_constraint_violation(violation));
            }
        }
        Ok(())
    }

    // ── Ownership ────────────────────────────────────────────────────────

    /// The name `SHOW CONSTRAINTS` gives the constraint on `(rel_type,
    /// property)`: the one its author registered, else the canonical descriptor.
    pub(crate) fn rel_unique_display_name(&self, rel_type: &str, property: &str) -> String {
        let properties = [property.to_string()];
        self.name_for_constraint(
            EntityKind::Relationship,
            rel_type,
            &properties,
            &[ConstraintKind::Unique],
        )
        .map(str::to_string)
        .unwrap_or_else(|| descriptor(rel_type, &properties))
    }

    /// Refuse to drop an index a constraint owns, naming the constraint.
    pub(crate) fn reject_owned_rel_index_drop(
        &self,
        rel_type: &str,
        property: &str,
    ) -> Result<(), String> {
        if !self.has_rel_unique_constraint(rel_type, property) {
            return Ok(());
        }
        let constraint = self.rel_unique_display_name(rel_type, property);
        Err(format!(
            "index 'relationship:{rel_type}.{property}' backs the constraint '{constraint}' and \
             cannot be dropped on its own. DROP CONSTRAINT {constraint} removes the constraint; \
             an index you declared yourself stays, one the constraint created goes with it."
        ))
    }

    /// Install the owned entry for every declared constraint after a load,
    /// adopting an index the persisted list names. A repeated value the loaded
    /// data already holds does not fail the load — a `.kgl` must always open.
    pub(super) fn rebuild_rel_unique_from_keys(&mut self) {
        let declared: Vec<IndexKey> = self.rel.unique.iter().cloned().collect();
        for key in declared {
            if self.rel.index_keys.contains(&key) {
                self.rel.adopted.insert(key.clone());
            }
            if !self.rel.indices.contains_key(&key) {
                self.build_rel_property_index(&key.0, &key.1);
            }
        }
    }

    /// Owned-only declared slots a deferred load has not built, for the
    /// listing.
    pub(super) fn deferred_owned_rel_slots(&self) -> Vec<IndexKey> {
        self.rel
            .unique
            .iter()
            .filter(|key| !self.rel.index_keys.contains(*key))
            .cloned()
            .collect()
    }

    /// The relationship index keys a save persists: every index except the
    /// owned-only ones, sorted.
    pub(super) fn persisted_rel_index_keys(&self) -> Vec<IndexKey> {
        let mut keys: Vec<IndexKey> = self
            .rel
            .indices
            .keys()
            .filter(|key| !self.rel.unique.contains(*key) || self.rel.adopted.contains(*key))
            .cloned()
            .collect();
        keys.sort_unstable();
        keys
    }
}

#[cfg(test)]
#[path = "rel_unique_tests.rs"]
mod tests;
