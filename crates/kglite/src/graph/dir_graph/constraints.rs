//! UNIQUE-constraint declaration and enforcement on `DirGraph`.
//!
//! # The structure *is* the constraint
//!
//! `DirGraph::unique_indices` maps `(node_type, [properties]) -> tuple value ->
//! the one NodeIndex occupying it`. Unlike the other secondary index kinds,
//! which map a value to a `Vec<NodeIndex>`, the unique index holds a single
//! occupant — so "is this tuple taken by somebody else?" is one hash lookup,
//! and there is no separate constraint bookkeeping that could drift out of sync
//! with the index that answers the question.
//!
//! # Cost per write
//!
//! `has_unique_constraints()` is one `HashMap::is_empty` — a graph that
//! declares no constraint (the default, and every graph saved before this
//! feature) pays exactly that on every write and allocates nothing. A type that
//! *does* carry constraints pays, per write: a scan of the declaration keys
//! (a handful, and the established pattern the other index kinds use), one
//! property read per constrained property, and one hash probe per constraint.
//! No full scan, and no per-write allocation beyond the claim vector, which is
//! only built when the type actually has a constraint.
//!
//! # NULL semantics
//!
//! A node is *exempt* from a unique constraint unless **every** property in the
//! tuple reads as present and non-null — matching Neo4j, where a uniqueness
//! constraint does not apply to nodes missing the property, and a composite
//! constraint requires the whole tuple. So many nodes may share "no email"
//! while `email` is still UNIQUE. [`Self::unique_claims`] encodes this by
//! producing no claim for an incomplete tuple.
//!
//! # Persistence
//!
//! `unique_indices` is `#[serde(skip)]`; the declaration list
//! `unique_constraint_keys` is persisted and replayed by
//! [`Self::rebuild_unique_indices_from_keys`] on load — the same
//! persist-keys/rebuild-on-load pattern the property, range, and composite
//! indexes use. The rebuild re-verifies the constraint for free, since building
//! a single-occupant map *is* the duplicate check.

use std::collections::{HashMap, HashSet};

use petgraph::graph::NodeIndex;

use super::indexes::PropertyReader;
use super::DirGraph;
use crate::datatypes::values::Value;
use crate::graph::constraints::{
    normalize_properties, ConstraintKind, ConstraintViolation, UniqueConstraintKey,
};
use crate::graph::schema::{CompositeValue, PROVISIONAL_KEY};

/// One node's occupancy of one declared unique tuple.
///
/// Produced by [`DirGraph::unique_claims`] before a write, checked by
/// [`DirGraph::check_unique_claims`], and redeemed by
/// [`DirGraph::commit_unique_claims`] once the node exists. Deriving the claim
/// once and reusing it for check + commit is what keeps the two from disagreeing
/// about *which* tuple was validated.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct UniqueClaim {
    pub key: UniqueConstraintKey,
    pub value: CompositeValue,
}

/// The unique-index bookkeeping a property write owes once it has been applied:
/// release what the node used to occupy, claim what it now occupies. Produced by
/// [`DirGraph::plan_property_write`] *before* the write (so the write can be
/// rejected without touching storage) and redeemed by
/// [`DirGraph::apply_property_write_plan`] after.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct PropertyWritePlan {
    pub release: Vec<UniqueClaim>,
    pub claim: Vec<UniqueClaim>,
}

impl DirGraph {
    // ========================================================================
    // Declaration
    // ========================================================================

    /// Whether *any* unique constraint is declared on this graph. The write
    /// path's fast-out: one `is_empty` for the overwhelmingly common case.
    #[inline]
    pub fn has_unique_constraints(&self) -> bool {
        !self.unique_indices.is_empty()
    }

    /// Whether `(node_type, properties)` is declared UNIQUE. Property order is
    /// irrelevant — `(a, b)` and `(b, a)` are the same constraint.
    pub fn has_unique_constraint(&self, node_type: &str, properties: &[String]) -> bool {
        self.find_unique_key(node_type, properties).is_some()
    }

    /// Every declared unique constraint, as `(node_type, properties)` in
    /// declaration order. Backs `SHOW CONSTRAINTS`.
    pub fn list_unique_constraints(&self) -> Vec<UniqueConstraintKey> {
        let mut all: Vec<UniqueConstraintKey> = self.unique_indices.keys().cloned().collect();
        all.sort();
        all
    }

    /// The stored declaration key matching `(node_type, properties)` up to
    /// property order, if declared.
    fn find_unique_key(
        &self,
        node_type: &str,
        properties: &[String],
    ) -> Option<&UniqueConstraintKey> {
        let wanted = normalize_properties(properties);
        self.unique_indices
            .keys()
            .find(|(nt, props)| nt == node_type && normalize_properties(props) == wanted)
    }

    /// Declare a UNIQUE constraint on `(node_type, properties)` and build its
    /// index from live data. Returns the number of distinct tuples indexed.
    ///
    /// Fails with [`ConstraintViolation::preexisting`] when the existing data
    /// already contains a duplicate — declaring a constraint the data violates
    /// would otherwise install a constraint that silently lies about the rows
    /// already present. Nothing is installed on failure.
    ///
    /// Idempotent: re-declaring an existing constraint rebuilds it rather than
    /// erroring, so `CREATE CONSTRAINT ... IF NOT EXISTS` and a reload both work.
    pub fn create_unique_constraint(
        &mut self,
        node_type: &str,
        properties: &[&str],
    ) -> Result<usize, ConstraintViolation> {
        if properties.is_empty() {
            // A constraint over no properties would claim one global slot and
            // reject the type's second node. Reject the declaration instead.
            return Err(ConstraintViolation::preexisting(
                ConstraintKind::Unique,
                node_type,
                Vec::new(),
                0,
                Vec::new(),
            ));
        }
        let owned: Vec<String> = properties.iter().map(|p| (*p).to_string()).collect();
        // Re-declaring replaces the previous spelling of the same constraint,
        // so the stored key reflects the latest declaration order.
        if let Some(existing) = self.find_unique_key(node_type, &owned).cloned() {
            self.remove_unique_declaration(&existing);
        }

        let key: UniqueConstraintKey = (node_type.to_string(), owned.clone());
        let (index, duplicates, sample) = self.build_unique_index(node_type, &owned);
        if duplicates > 0 {
            return Err(ConstraintViolation::preexisting(
                self.unique_kind_for(node_type, &owned),
                node_type,
                owned,
                duplicates,
                sample,
            ));
        }

        let count = index.len();
        self.unique_indices.insert(key.clone(), index);
        self.unique_constraint_keys.push(key);
        Ok(count)
    }

    /// Drop a declared unique constraint. Returns whether one was removed.
    pub fn drop_unique_constraint(&mut self, node_type: &str, properties: &[String]) -> bool {
        match self.find_unique_key(node_type, properties).cloned() {
            Some(key) => {
                self.remove_unique_declaration(&key);
                true
            }
            None => false,
        }
    }

    /// Forget a declaration in both the live index and the persisted key list.
    fn remove_unique_declaration(&mut self, key: &UniqueConstraintKey) {
        self.unique_indices.remove(key);
        self.unique_constraint_keys.retain(|stored| stored != key);
    }

    /// Drop every unique constraint declared on `node_type`. Used when the type
    /// itself goes away, so a later type of the same name does not inherit a
    /// constraint whose index refers to deleted nodes.
    pub fn drop_unique_constraints_for_type(&mut self, node_type: &str) -> usize {
        let keys: Vec<UniqueConstraintKey> = self
            .unique_indices
            .keys()
            .filter(|(nt, _)| nt == node_type)
            .cloned()
            .collect();
        for key in &keys {
            self.remove_unique_declaration(key);
        }
        keys.len()
    }

    /// Report `NODE KEY` when the tuple is exactly the type's declared primary
    /// key (which is unique *and* not null), `UNIQUE` otherwise. Derived rather
    /// than stored, so the two declarations cannot drift apart.
    fn unique_kind_for(&self, node_type: &str, properties: &[String]) -> ConstraintKind {
        match self.primary_key_for(node_type) {
            Some(pk) if properties.len() == 1 && properties[0] == pk => ConstraintKind::NodeKey,
            _ => ConstraintKind::Unique,
        }
    }

    /// Scan `node_type` and build the single-occupant map for `properties`.
    /// Returns `(index, duplicate_tuple_count, first_duplicate_sample)`.
    ///
    /// First occupant wins a contested tuple, so the returned index is always
    /// internally consistent even when the data is not — that is what lets the
    /// load path stay non-fatal (see
    /// [`Self::rebuild_unique_indices_from_keys`]).
    pub(super) fn build_unique_index(
        &mut self,
        node_type: &str,
        properties: &[String],
    ) -> (HashMap<CompositeValue, NodeIndex>, usize, Vec<Value>) {
        let readers: Vec<PropertyReader> = properties
            .iter()
            .map(|property| self.property_reader(node_type, property))
            .collect();

        let mut index: HashMap<CompositeValue, NodeIndex> = HashMap::new();
        let mut duplicates = 0usize;
        let mut sample: Vec<Value> = Vec::new();

        if let Some(node_indices) = self.type_indices.get(node_type) {
            for idx in node_indices.iter() {
                let Some(values) = self.read_complete_tuple(&readers, idx) else {
                    // Incomplete tuple — exempt, per the NULL semantics.
                    continue;
                };
                let composite = CompositeValue(values);
                if let Some(_occupant) = index.get(&composite) {
                    duplicates += 1;
                    if sample.is_empty() {
                        sample = composite.0.clone();
                    }
                    continue;
                }
                index.insert(composite, idx);
            }
        }

        (index, duplicates, sample)
    }

    /// Read every property of a constraint tuple off one node, returning `None`
    /// as soon as one is absent or null — an incomplete tuple is exempt.
    fn read_complete_tuple(
        &self,
        readers: &[PropertyReader],
        idx: NodeIndex,
    ) -> Option<Vec<Value>> {
        let mut values = Vec::with_capacity(readers.len());
        for reader in readers {
            match self.read_indexed(reader, idx) {
                Some(Value::Null) | None => return None,
                Some(value) => values.push(value),
            }
        }
        Some(values)
    }

    // ========================================================================
    // Write-path enforcement
    // ========================================================================

    /// The unique tuples a node of `node_type` would occupy, given a reader for
    /// its property values. `read` is called with the **user-facing** property
    /// name, matching how the constraint was declared; the caller decides where
    /// the value comes from (a pending CREATE's property map, a node already in
    /// the graph, a bulk row).
    ///
    /// Returns an empty vector when the type declares no constraint, or when
    /// every declared tuple is incomplete on this node — so the common case
    /// allocates nothing beyond an empty `Vec`.
    pub(crate) fn unique_claims<F>(&self, node_type: &str, read: F) -> Vec<UniqueClaim>
    where
        F: Fn(&str) -> Option<Value>,
    {
        if !self.has_unique_constraints() {
            return Vec::new();
        }
        let mut claims = Vec::new();
        for key in self.unique_indices.keys() {
            if key.0 != node_type {
                continue;
            }
            let mut values = Vec::with_capacity(key.1.len());
            let mut complete = true;
            for property in &key.1 {
                match read(property) {
                    Some(Value::Null) | None => {
                        complete = false;
                        break;
                    }
                    Some(value) => values.push(value),
                }
            }
            if complete {
                claims.push(UniqueClaim {
                    key: key.clone(),
                    value: CompositeValue(values),
                });
            }
        }
        claims
    }

    /// Reject the write if any claim's tuple is already occupied by a different
    /// node. `holder` is the node being written, when it already exists — a SET
    /// that rewrites a property to its current value must not conflict with
    /// itself.
    pub(crate) fn check_unique_claims(
        &self,
        claims: &[UniqueClaim],
        holder: Option<NodeIndex>,
    ) -> Result<(), ConstraintViolation> {
        for claim in claims {
            let Some(index) = self.unique_indices.get(&claim.key) else {
                continue;
            };
            match index.get(&claim.value) {
                Some(occupant) if Some(*occupant) != holder => {
                    return Err(ConstraintViolation::duplicate(
                        self.unique_kind_for(&claim.key.0, &claim.key.1),
                        claim.key.0.clone(),
                        claim.key.1.clone(),
                        claim.value.0.clone(),
                    ));
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// Record `node_idx` as the occupant of each claimed tuple. Call **after**
    /// the node exists and [`Self::check_unique_claims`] passed.
    pub(crate) fn commit_unique_claims(&mut self, claims: &[UniqueClaim], node_idx: NodeIndex) {
        for claim in claims {
            if let Some(index) = self.unique_indices.get_mut(&claim.key) {
                index.insert(claim.value.clone(), node_idx);
            }
        }
    }

    /// Give up `node_idx`'s occupancy of each claimed tuple — the old-value half
    /// of a SET, so the vacated tuple becomes available again. Only removes an
    /// entry this node actually holds.
    pub(crate) fn release_unique_claims(&mut self, claims: &[UniqueClaim], node_idx: NodeIndex) {
        for claim in claims {
            if let Some(index) = self.unique_indices.get_mut(&claim.key) {
                if index.get(&claim.value) == Some(&node_idx) {
                    index.remove(&claim.value);
                }
            }
        }
    }

    /// Evict deleted nodes from every unique index of `node_type`. Without this
    /// a deleted node keeps its tuple reserved forever and a legitimate re-insert
    /// of the same value would be rejected.
    ///
    /// O(distinct tuples) per constraint, matching the shape the delete path
    /// already uses for the property and composite indexes.
    pub(crate) fn evict_unique_claims_for_nodes(
        &mut self,
        node_type: &str,
        deleted: &HashSet<NodeIndex>,
    ) {
        if !self.has_unique_constraints() {
            return;
        }
        for (key, index) in self.unique_indices.iter_mut() {
            if key.0 != node_type {
                continue;
            }
            index.retain(|_, occupant| !deleted.contains(occupant));
        }
    }

    /// What a property write must do to the unique indexes once it has been
    /// applied: give up the tuples the node used to occupy, take the tuples it
    /// now occupies.
    ///
    /// Empty on both sides when the type declares no unique constraint, which is
    /// the common case.
    pub(crate) fn plan_property_write(
        &mut self,
        node_type: &str,
        node_idx: NodeIndex,
        property: &str,
        new_value: Option<&Value>,
    ) -> Result<PropertyWritePlan, ConstraintViolation> {
        let constrained = self.constrained_properties(node_type);
        if constrained.is_empty() {
            return Ok(PropertyWritePlan::default());
        }

        // One read pass over every property any constraint on this type cares
        // about, so the composite tuples can be rebuilt without re-reading.
        let mut before: HashMap<String, Value> = HashMap::with_capacity(constrained.len());
        for name in &constrained {
            let reader = self.property_reader(node_type, name);
            if let Some(value) = self.read_indexed(&reader, node_idx) {
                if !matches!(value, Value::Null) {
                    before.insert(name.clone(), value);
                }
            }
        }

        // The same map with this write applied — `None` models REMOVE and a SET
        // to null identically, since a constraint treats absent and null alike.
        let mut after = before.clone();
        match new_value {
            Some(Value::Null) | None => {
                after.remove(property);
            }
            Some(value) => {
                after.insert(property.to_string(), value.clone());
            }
        }

        // NOT NULL is evaluated against the post-write state, so a SET-to-null or
        // a REMOVE of a required property is caught even though the property is
        // present beforehand.
        self.check_required_fields(node_type, |name| after.get(name).cloned())?;

        let release = self.unique_claims(node_type, |name| before.get(name).cloned());
        let claim = self.unique_claims(node_type, |name| after.get(name).cloned());
        self.check_unique_claims(&claim, Some(node_idx))?;
        Ok(PropertyWritePlan { release, claim })
    }

    /// Every property name any declared constraint on `node_type` reads —
    /// unique tuples plus required fields, deduplicated.
    fn constrained_properties(&self, node_type: &str) -> Vec<String> {
        let mut names: Vec<String> = Vec::new();
        if self.has_unique_constraints() {
            for (nt, properties) in self.unique_indices.keys() {
                if nt == node_type {
                    names.extend(properties.iter().cloned());
                }
            }
        }
        names.extend(self.required_fields_for(node_type).iter().cloned());
        // The provisional marker decides whether NOT NULL applies at all, so it
        // has to be readable by the same composed map.
        if !names.is_empty() {
            names.push(PROVISIONAL_KEY.to_string());
        }
        names.sort();
        names.dedup();
        names
    }

    /// Apply a plan's index bookkeeping after the write landed.
    pub(crate) fn apply_property_write_plan(
        &mut self,
        plan: &PropertyWritePlan,
        node_idx: NodeIndex,
    ) {
        self.release_unique_claims(&plan.release, node_idx);
        self.commit_unique_claims(&plan.claim, node_idx);
    }

    /// The violation a second claim on `claim` raises, for a caller that detects
    /// the collision itself rather than through the stored index — the bulk path
    /// rejecting a repeat inside one input batch, where neither row is in the
    /// graph yet so there is no occupant to collide with.
    pub(crate) fn unique_batch_conflict(&self, claim: &UniqueClaim) -> ConstraintViolation {
        ConstraintViolation::duplicate(
            self.unique_kind_for(&claim.key.0, &claim.key.1),
            claim.key.0.clone(),
            claim.key.1.clone(),
            claim.value.0.clone(),
        )
    }

    // ========================================================================
    // NOT NULL (required fields)
    // ========================================================================

    /// The properties `node_type` declares as required, via
    /// `define_schema({"nodes": {"T": {"required": [...]}}})`. Empty when the
    /// type declares none.
    pub fn required_fields_for(&self, node_type: &str) -> &[String] {
        self.schema_definition
            .as_ref()
            .and_then(|schema| schema.node_schemas.get(node_type))
            .map(|node| node.required_fields.as_slice())
            .unwrap_or(&[])
    }

    /// Whether `node_type` declares any required field. The write-path fast-out:
    /// two `Option` hops and a length check, no allocation.
    #[inline]
    pub fn has_required_fields(&self, node_type: &str) -> bool {
        !self.required_fields_for(node_type).is_empty()
    }

    /// Reject a write that leaves a declared-required property absent or null.
    ///
    /// `read` is called with each required property name and returns its value
    /// *as the write will leave it* — the caller composes pending values over
    /// stored ones, so a SET that nulls a required property is caught even
    /// though the property is present beforehand.
    ///
    /// # Structural fields are exempt
    ///
    /// `id`, `title`, and `type` are always present by construction (they are
    /// `NodeData` fields, not properties), so requiring them is a no-op rather
    /// than an error — matching what the offline
    /// `mutation::validation::validate_single_node` checker has always done, so
    /// write-time and validation-time agree.
    ///
    /// # Provisional stubs are deferred, not exempt
    ///
    /// A write that carries `_provisional = true` is auto-vivification creating
    /// a placeholder for an edge endpoint whose real row has not arrived
    /// (`mutation::maintain::vivify_stubs`). Such a stub carries only its id by
    /// construction, so enforcing NOT NULL here would make `add_connections`
    /// fail on any edge list that mentions a node before its own row loads —
    /// i.e. it would break graph building outright.
    ///
    /// The escape hatch is the **existing promotion flow**, not an exemption
    /// flag: the stub is written, and the later `add_nodes` upsert that supplies
    /// the real row clears the `_provisional` marker
    /// (`mutation::batch::flush_chunk`) — and *that* write is a normal write, so
    /// it is fully enforced. A stub that is never promoted therefore never
    /// satisfies the constraint, and stays visible as one:
    /// `validate_schema()` reports it as a missing required field, and
    /// `purge_provisional_nodes()` (which the blueprint builder runs
    /// automatically) deletes it. So the incomplete state is bounded and
    /// reportable rather than silently blessed.
    ///
    /// Writing `_provisional = true` by hand therefore deliberately opts a node
    /// out of NOT NULL until it is promoted. The blueprint builder already
    /// refuses a spec that declares `_provisional` as a user property
    /// (`blueprint::build`), which is where a user would most plausibly do it by
    /// accident.
    pub(crate) fn check_required_fields<F>(
        &self,
        node_type: &str,
        read: F,
    ) -> Result<(), ConstraintViolation>
    where
        F: Fn(&str) -> Option<Value>,
    {
        let required = self.required_fields_for(node_type);
        if required.is_empty() {
            return Ok(());
        }
        if matches!(read(PROVISIONAL_KEY), Some(Value::Boolean(true))) {
            return Ok(());
        }
        for property in required {
            if matches!(property.as_str(), "id" | "title" | "type") {
                continue;
            }
            match read(property) {
                Some(Value::Null) | None => {
                    return Err(ConstraintViolation::missing(
                        self.required_kind_for(node_type, property),
                        node_type,
                        property,
                    ));
                }
                Some(_) => {}
            }
        }
        Ok(())
    }

    /// `NODE KEY` when the required property is also the type's primary key,
    /// `NOT NULL` otherwise — so one declaration does not report itself under
    /// two different names.
    fn required_kind_for(&self, node_type: &str, property: &str) -> ConstraintKind {
        match self.primary_key_for(node_type) {
            Some(pk) if pk == property => ConstraintKind::NodeKey,
            _ => ConstraintKind::NotNull,
        }
    }

    // ========================================================================
    // Load-time rebuild
    // ========================================================================

    /// Rebuild every persisted unique constraint from live data. Returns the
    /// violations found in the loaded data, if any.
    ///
    /// **Deliberately non-fatal.** A `.kgl` file must always open: refusing to
    /// load a graph because its data violates a constraint would strand the
    /// user's data behind the very tool they need to fix it. So a contested
    /// tuple keeps its first occupant, the constraint stays declared and live
    /// for all *subsequent* writes, and the duplicates are returned for the
    /// caller to surface. A violating file can only come from a write path that
    /// predates enforcement or one that bypasses it (see the RDF loaders in the
    /// coverage notes), not from a normal write.
    pub(crate) fn rebuild_unique_indices_from_keys(&mut self) -> Vec<ConstraintViolation> {
        let keys: Vec<UniqueConstraintKey> = std::mem::take(&mut self.unique_constraint_keys);
        let mut violations = Vec::new();
        for key in &keys {
            let (index, duplicates, sample) = self.build_unique_index(&key.0, &key.1);
            if duplicates > 0 {
                violations.push(ConstraintViolation::preexisting(
                    self.unique_kind_for(&key.0, &key.1),
                    key.0.clone(),
                    key.1.clone(),
                    duplicates,
                    sample,
                ));
            }
            self.unique_indices.insert(key.clone(), index);
        }
        self.unique_constraint_keys = keys;
        violations
    }

    /// Re-scan live data and report every unique-constraint violation currently
    /// present. The on-demand counterpart of the load-time rebuild, for callers
    /// that want to audit a graph filled by a path that bypasses enforcement.
    pub fn verify_unique_constraints(&mut self) -> Vec<ConstraintViolation> {
        let keys: Vec<UniqueConstraintKey> = self.unique_indices.keys().cloned().collect();
        let mut violations = Vec::new();
        for key in &keys {
            let (_, duplicates, sample) = self.build_unique_index(&key.0, &key.1);
            if duplicates > 0 {
                violations.push(ConstraintViolation::preexisting(
                    self.unique_kind_for(&key.0, &key.1),
                    key.0.clone(),
                    key.1.clone(),
                    duplicates,
                    sample,
                ));
            }
        }
        violations
    }
}
