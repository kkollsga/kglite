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
    normalize_properties, ConstraintKind, ConstraintViolation, NamedConstraint, UniqueConstraintKey,
};
use crate::graph::schema::{
    CompositeValue, NodeSchemaDefinition, SchemaDefinition, PROVISIONAL_KEY,
};

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
    pub(crate) fn has_unique_constraints(&self) -> bool {
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
    pub(crate) fn create_unique_constraint(
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
    pub(crate) fn drop_unique_constraint(
        &mut self,
        node_type: &str,
        properties: &[String],
    ) -> bool {
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

    /// Report `NODE KEY` when the tuple is unique *and* every property in it is
    /// required — which is what a node key is — and `UNIQUE` otherwise. Derived
    /// rather than stored, so the declarations cannot drift apart.
    ///
    /// The type's declared primary key satisfies this by construction. The
    /// second arm is what lets `CREATE CONSTRAINT … IS NODE KEY` report itself
    /// honestly: KGLite serves that statement as uniqueness plus presence, and
    /// there is only one primary-key slot per type, so a node key declared
    /// through DDL is not the primary key and would otherwise report as a plain
    /// `UNIQUE` violation.
    pub(crate) fn unique_kind_for(&self, node_type: &str, properties: &[String]) -> ConstraintKind {
        if matches!(self.primary_key_for(node_type), Some(pk) if properties.len() == 1 && properties[0] == pk)
        {
            return ConstraintKind::NodeKey;
        }
        let required = self.required_property_names(node_type);
        if !properties.is_empty()
            && properties
                .iter()
                .all(|property| required.contains(&property.as_str()))
        {
            return ConstraintKind::NodeKey;
        }
        ConstraintKind::Unique
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
    ///
    /// A rejection is *also* parked on the graph by
    /// [`DirGraph::record_constraint_violation`], so callers whose error channel
    /// is a `String` (the Cypher `SET` / `REMOVE` tree) still surface a typed
    /// `ConstraintViolationError`. Recording here rather than at each call site
    /// keeps the violation attached to the code that produced it.
    pub(crate) fn plan_property_write(
        &mut self,
        node_type: &str,
        node_idx: NodeIndex,
        property: &str,
        new_value: Option<&Value>,
    ) -> Result<PropertyWritePlan, ConstraintViolation> {
        let planned = self.plan_property_write_uncaught(node_type, node_idx, property, new_value);
        if let Err(violation) = &planned {
            self.record_constraint_violation(violation.clone());
        }
        planned
    }

    fn plan_property_write_uncaught(
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
        names.extend(
            self.required_property_names(node_type)
                .into_iter()
                .map(str::to_string),
        );
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
    pub(crate) fn required_fields_for(&self, node_type: &str) -> &[String] {
        self.schema_definition
            .as_ref()
            .and_then(|schema| schema.node_schemas.get(node_type))
            .map(|node| node.required_fields.as_slice())
            .unwrap_or(&[])
    }

    /// Whether `node_type` requires any property to be present. The write-path
    /// fast-out: a few `Option` hops, no allocation.
    ///
    /// True for a declared `primary_key` as well as for `required_fields`, since
    /// a primary key is unique **and** present (NODE KEY). A key on `id` does not
    /// count: `id` is a `NodeData` field that always exists.
    #[inline]
    pub(crate) fn has_required_fields(&self, node_type: &str) -> bool {
        if !self.required_fields_for(node_type).is_empty() {
            return true;
        }
        matches!(self.primary_key_for(node_type), Some(pk) if pk != "id")
    }

    /// Every property `node_type` requires to be present: the declared
    /// `required_fields` plus a non-`id` primary key. Borrow-free so the caller
    /// can hold `&self` while reading values.
    fn required_property_names(&self, node_type: &str) -> Vec<&str> {
        let mut names: Vec<&str> = self
            .required_fields_for(node_type)
            .iter()
            .map(String::as_str)
            .collect();
        // A primary key is required by definition. Skip `id` — it is a
        // `NodeData` field, present by construction.
        if let Some(pk) = self.primary_key_for(node_type) {
            if pk != "id" && !names.contains(&pk) {
                names.push(pk);
            }
        }
        names
    }

    /// Reject a write that leaves a declared-required property absent or null.
    ///
    /// `read` is called with each required property name and returns its value
    /// *as the write will leave it* — the caller composes pending values over
    /// stored ones, so a SET that nulls a required property is caught even
    /// though the property is present beforehand.
    ///
    /// # Structural fields: `type` is exempt, `id` and `title` are not
    ///
    /// `type` is the node's label rather than a value a write supplies, so no
    /// write can leave it absent and requiring it is a genuine no-op.
    ///
    /// `id` and `title` are **not** exempt, despite also being `NodeData`
    /// fields. Each write path resolves them before this check — CREATE
    /// auto-assigns an id and synthesizes a title from `name`/`title` or the
    /// label, and the bulk path falls back to the id column — so an omitted one
    /// reads as present and the requirement is satisfied. But both accept an
    /// *explicit* null (`CREATE (:T {title: null})`, `SET t.title = null`,
    /// `REMOVE t.title`, a null title cell in a batch), and the node that
    /// results genuinely carries a null. Skipping them here made
    /// `required: ["title"]` report itself through `SHOW CONSTRAINTS` as
    /// `NODE_PROPERTY_EXISTENCE` — and, with uniqueness alongside it, as
    /// `NODE_KEY` — while admitting exactly those writes: a constraint that
    /// reported success and enforced nothing, the one outcome this module
    /// refuses everywhere else (see `reject_structural_uniqueness`).
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
        let required = self.required_property_names(node_type);
        if required.is_empty() {
            return Ok(());
        }
        if matches!(read(PROVISIONAL_KEY), Some(Value::Boolean(true))) {
            return Ok(());
        }
        for property in required {
            if property == "type" {
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
    // NOT NULL declaration
    // ========================================================================

    /// Declare `property` NOT NULL on `node_type` — i.e. add it to the type's
    /// `required_fields`, the list [`Self::check_required_fields`] enforces on
    /// every write path.
    ///
    /// Returns the number of nodes of the type that were checked.
    ///
    /// Fails with [`ConstraintViolation::preexisting_missing`] when existing
    /// nodes have no value for the property, and installs nothing in that case —
    /// mirroring [`Self::create_unique_constraint`], because a constraint that
    /// silently exempts the rows already present is worse than a rejected
    /// declaration. Provisional stubs are skipped, matching the write-path rule:
    /// a stub is *deferred*, not exempt, and stays reportable via
    /// `validate_schema()`.
    ///
    /// Idempotent: re-declaring an existing requirement re-verifies it and
    /// changes nothing, so `IF NOT EXISTS` and a reload both work.
    pub(crate) fn create_not_null_constraint(
        &mut self,
        node_type: &str,
        property: &str,
    ) -> Result<usize, ConstraintViolation> {
        let (checked, missing) = self.count_missing_property(node_type, property);
        if missing > 0 {
            return Err(ConstraintViolation::preexisting_missing(
                self.required_kind_for(node_type, property),
                node_type,
                property,
                missing,
            ));
        }
        self.ddl_not_null_constraints
            .insert((node_type.to_string(), property.to_string()));
        self.require_property(node_type, property);
        Ok(checked)
    }

    /// Add `property` to `node_type`'s `required_fields`, keeping the list
    /// sorted and duplicate-free. The storage half of a presence declaration,
    /// shared by the DDL entry point and by [`Self::reapply_ddl_not_null`].
    fn require_property(&mut self, node_type: &str, property: &str) {
        let required = &mut self.node_schema_mut(node_type).required_fields;
        required.push(property.to_string());
        required.sort();
        required.dedup();
    }

    /// Re-add every DDL-declared presence constraint to the schema now installed.
    ///
    /// `required_fields` lives inside the `SchemaDefinition`, so installing a
    /// schema replaces the list a `CREATE CONSTRAINT ... IS NOT NULL` wrote into
    /// — silently un-enforcing it. The uniqueness half has no such problem: its
    /// index lives outside the schema and `set_schema` withdraws only what the
    /// *outgoing schema* declared. This restores the symmetry, so a DDL
    /// constraint is withdrawn only by `DROP CONSTRAINT`.
    pub(crate) fn reapply_ddl_not_null(&mut self) {
        if self.ddl_not_null_constraints.is_empty() {
            return;
        }
        let declared: Vec<(String, String)> =
            self.ddl_not_null_constraints.iter().cloned().collect();
        for (node_type, property) in declared {
            self.require_property(&node_type, &property);
        }
    }

    /// Withdraw a NOT NULL declaration. Reports whether one was removed.
    pub(crate) fn drop_not_null_constraint(&mut self, node_type: &str, property: &str) -> bool {
        // Forget the DDL provenance first, or the next schema install would
        // reinstate what was just dropped.
        self.ddl_not_null_constraints
            .remove(&(node_type.to_string(), property.to_string()));
        let Some(node) = self
            .schema_definition
            .as_mut()
            .and_then(|schema| schema.node_schemas.get_mut(node_type))
        else {
            return false;
        };
        let before = node.required_fields.len();
        node.required_fields.retain(|field| field != property);
        before != node.required_fields.len()
    }

    /// Whether `property` is declared NOT NULL on `node_type`. A non-`id`
    /// primary key counts: it is required by definition.
    pub(crate) fn has_not_null_constraint(&self, node_type: &str, property: &str) -> bool {
        self.required_property_names(node_type).contains(&property)
    }

    /// Every declared presence constraint, as `(node_type, property)` sorted.
    /// Backs `SHOW CONSTRAINTS` together with
    /// [`Self::list_unique_constraints`].
    pub(crate) fn list_not_null_constraints(&self) -> Vec<(String, String)> {
        let Some(schema) = self.schema_definition.as_ref() else {
            return Vec::new();
        };
        let mut all: Vec<(String, String)> = schema
            .node_schemas
            .keys()
            .flat_map(|node_type| {
                self.required_property_names(node_type)
                    .into_iter()
                    .map(move |property| (node_type.clone(), property.to_string()))
            })
            .collect();
        all.sort();
        all.dedup();
        all
    }

    /// `(nodes_checked, nodes_missing_the_property)` for `node_type`.
    /// Provisional stubs are skipped — see [`Self::check_required_fields`].
    fn count_missing_property(&mut self, node_type: &str, property: &str) -> (usize, usize) {
        // `type` is the node's label rather than a supplied value, so nothing
        // can be missing. `id`/`title` are read through `read_indexed` like any
        // other property — they are resolved on every write path but can be
        // explicitly nulled, so existing nulls must block the declaration.
        if property == "type" {
            let checked = self
                .type_indices
                .get(node_type)
                .map_or(0, |nodes| nodes.iter().count());
            return (checked, 0);
        }
        let reader = self.property_reader(node_type, property);
        let provisional = self.property_reader(node_type, PROVISIONAL_KEY);
        let Some(node_indices) = self.type_indices.get(node_type) else {
            return (0, 0);
        };
        let indices: Vec<NodeIndex> = node_indices.iter().collect();
        let mut missing = 0usize;
        for idx in &indices {
            if matches!(
                self.read_indexed(&provisional, *idx),
                Some(Value::Boolean(true))
            ) {
                continue;
            }
            match self.read_indexed(&reader, *idx) {
                Some(Value::Null) | None => missing += 1,
                Some(_) => {}
            }
        }
        (indices.len(), missing)
    }

    /// The mutable `NodeSchemaDefinition` for `node_type`, creating the schema
    /// container and the per-type entry when they do not exist yet.
    ///
    /// Declaring a constraint on a graph with no `define_schema` call is
    /// legitimate — `CREATE CONSTRAINT` is exactly that — so this materializes
    /// the schema rather than refusing. It deliberately does **not** go through
    /// `set_schema`, which installs the unique constraints a *whole* schema
    /// implies; presence constraints install no index, and the caller owns the
    /// uniqueness half of a NODE KEY.
    fn node_schema_mut(&mut self, node_type: &str) -> &mut NodeSchemaDefinition {
        self.schema_definition
            .get_or_insert_with(SchemaDefinition::new)
            .node_schemas
            .entry(node_type.to_string())
            .or_default()
    }

    // ========================================================================
    // Named constraints
    // ========================================================================

    /// Record the name its author gave a constraint, so
    /// `DROP CONSTRAINT <name>` can find it. Replaces any previous registration
    /// of the same name.
    pub(crate) fn register_constraint_name(&mut self, name: &str, constraint: NamedConstraint) {
        self.constraint_names.insert(name.to_string(), constraint);
    }

    /// The declaration registered under `name`, if any.
    pub(crate) fn constraint_by_name(&self, name: &str) -> Option<&NamedConstraint> {
        self.constraint_names.get(name)
    }

    /// Forget a name. Called when its constraint is dropped.
    pub(crate) fn forget_constraint_name(&mut self, name: &str) {
        self.constraint_names.remove(name);
    }

    /// The name registered for `(node_type, properties)`, if the constraint was
    /// declared with one. Property order is irrelevant, matching constraint
    /// identity. Lets `SHOW CONSTRAINTS` report the author's name.
    ///
    /// Several names can point at one tuple — `CREATE CONSTRAINT u … IS UNIQUE`
    /// and `CREATE CONSTRAINT nn … IS NOT NULL` on the same property are two
    /// declarations that `SHOW CONSTRAINTS` reports as a single `NODE_KEY` row.
    /// Picking the first match out of a `HashMap` made *which* name that row
    /// carried depend on hash order, so the same graph reported different names
    /// before and after a save/load round-trip. The lowest name in sort order
    /// wins instead: arbitrary, but stable across runs, processes and reloads,
    /// which is what a listing an operator reads during a migration needs.
    pub(crate) fn name_for_constraint(
        &self,
        node_type: &str,
        properties: &[String],
    ) -> Option<&str> {
        let wanted = normalize_properties(properties);
        self.constraint_names
            .iter()
            .filter(|(_, declared)| {
                declared.node_type == node_type
                    && normalize_properties(&declared.properties) == wanted
            })
            .map(|(name, _)| name.as_str())
            .min()
    }

    /// Drop every registered name whose constraint is no longer declared.
    ///
    /// The registry is a lookup aid, not the source of truth, and several paths
    /// remove a constraint without going through `DROP CONSTRAINT` —
    /// [`Self::drop_unique_constraints_for_type`] when a type is deleted, and
    /// `set_schema` replacing a schema that declared one. Without this, those
    /// names would leak into every subsequent save and `DROP CONSTRAINT <name>`
    /// would claim to drop something that had already gone.
    pub(crate) fn prune_constraint_names(&mut self) {
        if self.constraint_names.is_empty() {
            return;
        }
        let live: Vec<String> = self
            .constraint_names
            .iter()
            .filter(|(_, declared)| self.constraint_is_declared(declared))
            .map(|(name, _)| name.clone())
            .collect();
        self.constraint_names.retain(|name, _| live.contains(name));
    }

    /// Whether the declaration a registered name points at is still in force.
    fn constraint_is_declared(&self, declared: &NamedConstraint) -> bool {
        match declared.kind {
            ConstraintKind::Unique => {
                self.has_unique_constraint(&declared.node_type, &declared.properties)
            }
            ConstraintKind::NotNull => declared
                .properties
                .iter()
                .all(|property| self.has_not_null_constraint(&declared.node_type, property)),
            // A NODE KEY is the conjunction, so it survives only while both
            // halves do — a dropped uniqueness half demotes it, and reporting it
            // as still declared would overstate what is enforced.
            ConstraintKind::NodeKey => {
                self.has_unique_constraint(&declared.node_type, &declared.properties)
                    && declared
                        .properties
                        .iter()
                        .all(|property| self.has_not_null_constraint(&declared.node_type, property))
            }
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

    /// Recompute the unique-occupancy maps of every declared constraint on
    /// `node_types`, from live data.
    ///
    /// The statement-rollback counterpart of
    /// [`Self::rebuild_unique_indices_from_keys`]. `unique_indices` is parked by
    /// `rollback::swap_data_scale`, so a journal rollback leaves the *failed
    /// statement's* occupancy in place while the data underneath is restored;
    /// the claims the statement added or released have to be recomputed, or the
    /// graph keeps a phantom occupant (a permanent spurious
    /// `ConstraintViolationError` for a value nothing holds) or has silently
    /// released one (a real duplicate admitted on the next write).
    ///
    /// Scoped to the types the replay touched, so an untouched or unconstrained
    /// type costs nothing and an unconstrained graph returns immediately.
    /// Duplicates are not reported: the restored data is the pre-statement data,
    /// which the write path already accepted, so a contested tuple here could
    /// only be a pre-existing violation the load path already surfaced.
    pub(super) fn rebuild_unique_indices_for_types(&mut self, node_types: &HashSet<String>) {
        if node_types.is_empty() || self.unique_indices.is_empty() {
            return;
        }
        let keys: Vec<UniqueConstraintKey> = self
            .unique_indices
            .keys()
            .filter(|(node_type, _)| node_types.contains(node_type))
            .cloned()
            .collect();
        for key in keys {
            let (index, _duplicates, _sample) = self.build_unique_index(&key.0, &key.1);
            self.unique_indices.insert(key, index);
        }
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

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod not_null_declaration_tests {
    use super::*;
    use crate::graph::schema::NodeData;
    use crate::graph::storage::GraphWrite;

    /// `Person` nodes carrying whichever properties each row supplies, so a row
    /// can deliberately omit one.
    fn person_graph(rows: &[(i64, &str, Option<&str>)]) -> DirGraph {
        let mut graph = DirGraph::new();
        for (id, name, email) in rows {
            let mut props =
                HashMap::from([("name".to_string(), Value::String((*name).to_string()))]);
            if let Some(email) = email {
                props.insert("email".to_string(), Value::String((*email).to_string()));
            }
            let node = NodeData::new(
                Value::UniqueId(*id as u32),
                Value::String((*name).to_string()),
                "Person".to_string(),
                props,
                &mut graph.interner,
            );
            let idx = graph.graph.add_node(node);
            graph
                .type_indices
                .entry_or_default("Person".to_string())
                .push(idx);
        }
        graph
    }

    #[test]
    fn declaring_not_null_on_clean_data_installs_and_enforces_it() {
        let mut graph = person_graph(&[(1, "Alice", Some("a@b.c")), (2, "Bob", Some("b@b.c"))]);
        assert_eq!(
            graph.create_not_null_constraint("Person", "email").unwrap(),
            2
        );
        assert!(graph.has_not_null_constraint("Person", "email"));
        assert!(graph.has_required_fields("Person"));

        // A write that omits the property is now rejected.
        let violation = graph
            .check_required_fields("Person", |_| None)
            .expect_err("a write with no email must be rejected");
        assert_eq!(violation.kind, ConstraintKind::NotNull);
        assert!(violation.to_string().contains("'email'"));

        // A write that supplies it passes.
        graph
            .check_required_fields("Person", |name| {
                (name == "email").then(|| Value::String("c@b.c".to_string()))
            })
            .expect("a write with an email must pass");
    }

    #[test]
    fn declaring_not_null_against_missing_values_is_rejected_and_changes_nothing() {
        let mut graph = person_graph(&[(1, "Alice", Some("a@b.c")), (2, "Bob", None)]);
        let violation = graph
            .create_not_null_constraint("Person", "email")
            .expect_err("one node has no email");

        assert!(violation.is_declaration_failure());
        let message = violation.to_string();
        assert!(message.contains("cannot declare"), "{message}");
        assert!(message.contains("1 existing node"), "{message}");

        // Nothing installed: the graph must be as permissive as before.
        assert!(!graph.has_not_null_constraint("Person", "email"));
        assert!(!graph.has_required_fields("Person"));
    }

    #[test]
    fn declaring_not_null_is_idempotent() {
        let mut graph = person_graph(&[(1, "Alice", Some("a@b.c"))]);
        graph.create_not_null_constraint("Person", "email").unwrap();
        graph.create_not_null_constraint("Person", "email").unwrap();
        assert_eq!(
            graph.list_not_null_constraints(),
            vec![("Person".to_string(), "email".to_string())]
        );
    }

    #[test]
    fn dropping_not_null_stops_enforcement_and_reports_whether_it_existed() {
        let mut graph = person_graph(&[(1, "Alice", Some("a@b.c"))]);
        graph.create_not_null_constraint("Person", "email").unwrap();

        assert!(graph.drop_not_null_constraint("Person", "email"));
        assert!(!graph.has_not_null_constraint("Person", "email"));
        graph
            .check_required_fields("Person", |_| None)
            .expect("no requirement remains");

        // A second drop has nothing to remove.
        assert!(!graph.drop_not_null_constraint("Person", "email"));
        assert!(!graph.drop_not_null_constraint("Person", "nonexistent"));
    }

    /// A provisional stub carries only its id by construction, so counting it as
    /// a missing value would make declaring NOT NULL impossible on any graph
    /// built from an edge list. Deferred, not exempt — see
    /// `check_required_fields`.
    #[test]
    fn provisional_stubs_do_not_block_a_declaration() {
        let mut graph = person_graph(&[(1, "Alice", Some("a@b.c"))]);
        let stub = NodeData::new(
            Value::UniqueId(2),
            Value::String("stub".to_string()),
            "Person".to_string(),
            HashMap::from([(PROVISIONAL_KEY.to_string(), Value::Boolean(true))]),
            &mut graph.interner,
        );
        let idx = graph.graph.add_node(stub);
        graph
            .type_indices
            .entry_or_default("Person".to_string())
            .push(idx);

        graph
            .create_not_null_constraint("Person", "email")
            .expect("the stub must not block the declaration");
        assert!(graph.has_not_null_constraint("Person", "email"));
    }

    /// `type` is the node's label rather than a supplied value, so requiring it
    /// is satisfied by construction — nothing a write does can leave it absent.
    #[test]
    fn declaring_not_null_on_the_label_field_is_satisfied() {
        let mut graph = person_graph(&[(1, "Alice", None)]);
        assert_eq!(
            graph.create_not_null_constraint("Person", "type").unwrap(),
            1
        );
        graph
            .check_required_fields("Person", |_| None)
            .expect("type is the label and always present");
    }

    /// `id`/`title` are `NodeData` fields too, but a write can null them
    /// explicitly, so they are enforced rather than exempt. Every write path
    /// resolves them before the check (CREATE auto-assigns an id and synthesizes
    /// a title), which is what the reader models here: a resolved value passes,
    /// an unresolved one is the explicit-null case and must be rejected.
    /// Skipping them made `required: ["title"]` report itself through
    /// `SHOW CONSTRAINTS` while admitting `CREATE (:T {title: null})`.
    #[test]
    fn declaring_not_null_on_id_or_title_is_enforced_against_an_explicit_null() {
        for property in ["id", "title"] {
            let mut graph = person_graph(&[(1, "Alice", None)]);
            assert_eq!(
                graph
                    .create_not_null_constraint("Person", property)
                    .unwrap(),
                1
            );

            graph
                .check_required_fields("Person", |name| {
                    (name == property).then(|| Value::String("resolved".to_string()))
                })
                .expect("a write that carries the structural field passes");

            let violation = graph
                .check_required_fields("Person", |_| None)
                .expect_err("an explicitly nulled structural field must be rejected");
            assert_eq!(violation.kind, ConstraintKind::NotNull);
            assert!(violation.to_string().contains(property), "{violation}");
        }
    }

    #[test]
    fn declaring_not_null_needs_no_prior_define_schema() {
        let mut graph = person_graph(&[(1, "Alice", Some("a@b.c"))]);
        assert!(graph.get_schema().is_none());
        graph.create_not_null_constraint("Person", "email").unwrap();
        assert!(graph.get_schema().is_some());
    }

    /// A tuple that is both unique and fully required *is* a node key, so it
    /// must report itself as one — otherwise `CREATE CONSTRAINT … IS NODE KEY`
    /// would raise `UNIQUE` violations for a constraint the user declared as a
    /// node key.
    #[test]
    fn unique_plus_required_reports_as_node_key() {
        let mut graph = person_graph(&[(1, "Alice", Some("a@b.c"))]);
        graph
            .create_unique_constraint("Person", &["email"])
            .unwrap();
        assert_eq!(
            graph.unique_kind_for("Person", &["email".to_string()]),
            ConstraintKind::Unique
        );

        graph.create_not_null_constraint("Person", "email").unwrap();
        assert_eq!(
            graph.unique_kind_for("Person", &["email".to_string()]),
            ConstraintKind::NodeKey
        );
    }
}

#[cfg(test)]
mod constraint_name_registry_tests {
    use super::*;

    fn named(kind: ConstraintKind, properties: &[&str]) -> NamedConstraint {
        NamedConstraint {
            kind,
            node_type: "Person".to_string(),
            properties: properties.iter().map(|p| (*p).to_string()).collect(),
        }
    }

    #[test]
    fn a_registered_name_resolves_to_its_declaration() {
        let mut graph = DirGraph::new();
        graph
            .create_unique_constraint("Person", &["email"])
            .unwrap();
        graph.register_constraint_name(
            "person_email_unique",
            named(ConstraintKind::Unique, &["email"]),
        );

        assert_eq!(
            graph.constraint_by_name("person_email_unique"),
            Some(&named(ConstraintKind::Unique, &["email"]))
        );
        assert_eq!(graph.constraint_by_name("nope"), None);
        assert_eq!(
            graph.name_for_constraint("Person", &["email".to_string()]),
            Some("person_email_unique")
        );
    }

    /// Constraint identity ignores property order, so a name lookup must too.
    #[test]
    fn a_name_resolves_regardless_of_property_order() {
        let mut graph = DirGraph::new();
        graph
            .create_unique_constraint("Person", &["first", "last"])
            .unwrap();
        graph.register_constraint_name(
            "full_name",
            named(ConstraintKind::Unique, &["first", "last"]),
        );
        assert_eq!(
            graph.name_for_constraint("Person", &["last".to_string(), "first".to_string()]),
            Some("full_name")
        );
    }

    #[test]
    fn forgetting_a_name_leaves_the_constraint_in_force() {
        let mut graph = DirGraph::new();
        graph
            .create_unique_constraint("Person", &["email"])
            .unwrap();
        graph.register_constraint_name("c", named(ConstraintKind::Unique, &["email"]));

        graph.forget_constraint_name("c");
        assert_eq!(graph.constraint_by_name("c"), None);
        assert!(graph.has_unique_constraint("Person", &["email".to_string()]));
    }

    /// The registry is a lookup aid, so a name whose declaration went away by
    /// another route must not survive into the next save.
    #[test]
    fn pruning_discards_names_whose_declaration_is_gone() {
        let mut graph = DirGraph::new();
        graph
            .create_unique_constraint("Person", &["email"])
            .unwrap();
        graph.register_constraint_name("live", named(ConstraintKind::Unique, &["email"]));
        graph.register_constraint_name("dangling", named(ConstraintKind::Unique, &["nickname"]));

        graph.prune_constraint_names();
        assert!(graph.constraint_by_name("live").is_some());
        assert!(
            graph.constraint_by_name("dangling").is_none(),
            "a name with no declaration behind it must be pruned"
        );

        // Dropping the type's constraints strands the surviving name too.
        graph.drop_unique_constraints_for_type("Person");
        graph.prune_constraint_names();
        assert!(graph.constraint_by_name("live").is_none());
    }

    /// A NODE KEY is uniqueness *and* presence, so losing either half must
    /// demote it rather than leave the name claiming both are enforced.
    #[test]
    fn a_node_key_name_is_pruned_when_either_half_goes() {
        let mut graph = DirGraph::new();
        graph
            .create_unique_constraint("Person", &["email"])
            .unwrap();
        graph.create_not_null_constraint("Person", "email").unwrap();
        graph.register_constraint_name("person_key", named(ConstraintKind::NodeKey, &["email"]));

        graph.prune_constraint_names();
        assert!(graph.constraint_by_name("person_key").is_some());

        graph.drop_not_null_constraint("Person", "email");
        graph.prune_constraint_names();
        assert!(
            graph.constraint_by_name("person_key").is_none(),
            "a NODE KEY without its presence half is no longer a NODE KEY"
        );
    }

    #[test]
    fn populate_index_keys_prunes_and_sorts_deterministically() {
        let mut graph = DirGraph::new();
        graph
            .create_unique_constraint("Person", &["email"])
            .unwrap();
        graph
            .create_unique_constraint("Person", &["nickname"])
            .unwrap();
        graph.register_constraint_name("dangling", named(ConstraintKind::Unique, &["absent"]));

        graph.populate_index_keys();
        assert!(graph.constraint_by_name("dangling").is_none());
        // Sorted, so a graph carrying constraints saves reproducible bytes.
        let mut sorted = graph.unique_constraint_keys.clone();
        sorted.sort();
        assert_eq!(graph.unique_constraint_keys, sorted);
    }
}
