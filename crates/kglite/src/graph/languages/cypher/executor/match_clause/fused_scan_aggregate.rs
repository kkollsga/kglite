// Parallel execution of the fused node-scan aggregate: the partition state,
// the partition-invariant compiled context, and the two drivers (sequential
// and fanned-out) that share the per-node loop.
//
// Split out of `fused_match.rs` when that file crossed the 2500-line
// source-quality ceiling. `include!`d into `match_clause.rs` alongside its
// sibling, so these are still inherent methods on `CypherExecutor` with the
// same imports — the split is a file boundary, not a module boundary.

/// Per-partition scan state: the groups a partition discovered, in the order
/// it first saw them, with one accumulator each.
///
/// Partitions are contiguous index ranges of the candidate vector, so merging
/// them **in partition order** reproduces the sequential scan's first-seen
/// group order exactly — including which node each group carries as its
/// binding for the unsupported-aggregate fallback.
#[derive(Default)]
struct ScanPartial {
    groups: Vec<(Vec<Value>, NodeIndex)>,
    accumulators: Vec<InlineAccumulators>,
    index: FxHashMap<Vec<Value>, usize>,
}

impl ScanPartial {
    /// Fold a later partition into this one. `other` must come after `self` in
    /// candidate order for the emission order to match the sequential scan.
    fn merge_from(&mut self, other: ScanPartial) {
        for ((key, first_idx), acc) in other.groups.into_iter().zip(other.accumulators) {
            match self.index.get(&key) {
                Some(&gi) => self.accumulators[gi].merge(acc),
                None => {
                    self.index.insert(key.clone(), self.groups.len());
                    self.groups.push((key, first_idx));
                    self.accumulators.push(acc);
                }
            }
        }
    }
}

/// The partition-invariant half of a fused node-scan aggregate: the compiled
/// trees, which are immutable and shared across partitions.
struct ScanPartitionCtx<'c, 'q> {
    node_var: &'c str,
    compiled_where: Option<&'c ScanPred<'q>>,
    compiled_group: &'c [ScanExpr<'q>],
    compiled_agg: &'c [Option<ScanExpr<'q>>],
    agg_is_distinct: &'c [bool],
    needs_node: bool,
}

impl ScanPartitionCtx<'_, '_> {
    /// Which side of the runtime gate this scan's per-row work sits on. A
    /// fully compiled scan reads a column and does arithmetic (tens of ns a
    /// row) and needs a lot of rows to repay a fan-out; anything that falls
    /// back to `evaluate_expression` costs enough per row to repay it much
    /// sooner.
    fn cost_class(&self) -> parallel::CostClass {
        let compiled = self.compiled_where.is_none_or(ScanPred::is_compiled)
            && self.compiled_group.iter().all(ScanExpr::is_compiled)
            && self
                .compiled_agg
                .iter()
                .all(|arg| arg.as_ref().is_none_or(ScanExpr::is_compiled));
        if compiled {
            parallel::CostClass::Compiled
        } else {
            parallel::CostClass::Interpreted
        }
    }
}

/// Partitions per worker. More than one so a partition that happens to hold
/// the expensive rows cannot stall the whole scan, few enough that the
/// per-partition group maps stay a rounding error against the scan itself.
const SCAN_PARTITIONS_PER_WORKER: usize = 4;

impl<'g> CypherExecutor<'g> {
    /// Scan one contiguous range of candidates into a [`ScanPartial`].
    ///
    /// This is the whole per-node loop of the fused node-scan aggregate; the
    /// sequential and parallel drivers differ only in how many of these they
    /// run and on which threads. `runtime` is the partition's own — the route
    /// table and column-store handle it memoises are per-node-type mutable
    /// state that must not be shared.
    fn scan_partition<'r, F>(
        &self,
        nodes: &[NodeIndex],
        ctx: &ScanPartitionCtx<'_, '_>,
        runtime: &mut ScanRuntime<'r>,
        interrupt: &ParallelInterrupt<F>,
    ) -> Result<ScanPartial, String>
    where
        F: Fn() -> Option<String> + Sync,
        'g: 'r,
    {
        // One reusable ResultRow for the whole partition — no per-node alloc.
        let mut eval_row = ResultRow::new();
        eval_row
            .node_bindings
            .insert(ctx.node_var.to_string(), NodeIndex::new(0));

        let mut partial = ScanPartial::default();
        // Perf: reusable per-row scratch buffers — avoids a heap allocation per
        // passing row for the group key and aggregate-input vectors (the inner
        // loop's dominant cost on scan-heavy filters/aggregates).
        let mut key_values: Vec<Value> = Vec::with_capacity(ctx.compiled_group.len());
        let mut agg_vals: Vec<Value> = Vec::with_capacity(ctx.compiled_agg.len());

        for (scan_count, &node_idx) in nodes.iter().enumerate() {
            interrupt.check(scan_count)?;
            // Set the node binding for expression evaluation
            *eval_row
                .node_bindings
                .get_mut(ctx.node_var)
                .expect("invariant: node_var binding inserted upstream by pattern match") =
                node_idx;

            // One view per node — the store handle and the property routes are
            // memoised by node type.
            let node = if ctx.needs_node {
                runtime.bind(self.graph, node_idx)
            } else {
                None
            };

            // Check WHERE predicate. A predicate that cannot be evaluated
            // does not match (the row is dropped, not raised) — except for an
            // uncompilable regex, which the unfused path raises. See
            // `ScanPred::keeps_row`.
            if let Some(pred) = ctx.compiled_where {
                if !pred.keeps_row(self, runtime, node, &eval_row)? {
                    continue;
                }
            }

            // Evaluate group key (reuse the scratch buffer — no per-row
            // alloc). Errors propagate — same contract as the materialized
            // aggregation path: null groups arrive as Ok(Null); an Err is a
            // genuine error (missing parameter, overflow, …), never a group.
            key_values.clear();
            for expr in ctx.compiled_group {
                key_values.push(expr.eval(self, runtime, node, &eval_row)?);
            }

            // Evaluate all aggregate inputs for this node (reuse buffer).
            // Errors propagate — same contract as the group keys above and the
            // materialized aggregation path: a null argument arrives as
            // Ok(Null) and is skipped by the accumulators; an Err is a genuine
            // error (missing parameter, overflow, …).
            agg_vals.clear();
            for compiled in ctx.compiled_agg {
                agg_vals.push(match compiled {
                    // count(*) / count(<bound var>) marker — always counted.
                    None => Value::Boolean(true),
                    Some(expr) => expr.eval(self, runtime, node, &eval_row)?,
                });
            }

            let group_idx = match partial.index.get(&key_values) {
                Some(&group_idx) => group_idx,
                None => {
                    let group_idx = partial.groups.len();
                    partial.index.insert(key_values.clone(), group_idx);
                    partial.groups.push((key_values.clone(), node_idx));
                    partial
                        .accumulators
                        .push(InlineAccumulators::new(ctx.agg_is_distinct));
                    group_idx
                }
            };
            let acc = &mut partial.accumulators[group_idx];
            for (ai, val) in agg_vals.iter().enumerate() {
                acc.absorb(ai, val, ctx.agg_is_distinct[ai]);
            }
        }
        Ok(partial)
    }

    /// Whether this scan may fan out: the caller opted in, the region is
    /// provably free of shared-cache writes, and the runtime gate says there
    /// is enough work.
    ///
    /// The write-freedom argument (D4) is structural rather than a list of
    /// pre-warms, because a `Generic` arm can re-enter the whole interpreter
    /// and no enumeration of what it might touch stays true. Instead:
    ///
    /// * **Disk is excluded** — its `node_weight` materialises into the shared
    ///   query arena on every call. A `parallel` request on a disk graph is
    ///   silently served serially, because `parallel` is a hint and refusing
    ///   it would break portable code that runs against all three modes.
    /// * **Spatial graphs are excluded** — a spatial config disables the scan
    ///   compiler outright, so every row routes through the interpreter and
    ///   populates the per-node spatial cache. That is a genuine per-row
    ///   `&self` cache write with nothing to pre-warm.
    /// * Everything a **compiled** tree can reach — `node_weight`,
    ///   `column_store`, `PropRoute::resolve` — is a plain map read on the
    ///   memory, forked and mapped backends.
    ///
    /// What is left, on an interpreted arm, are lazily built caches that are
    /// all lock-guarded and first-writer-wins (the id index, memory peer
    /// counts, the mapped type/property indexes, the process-global regex
    /// cache). Those are race-*correct*; the cost of losing the race is a
    /// duplicated build on first touch, not a wrong answer.
    fn may_fan_out_scan(&self, rows: usize, ctx: &ScanPartitionCtx<'_, '_>) -> bool {
        self.parallel
            && !self.graph.graph.is_disk()
            && self.graph.spatial_configs.is_empty()
            && parallel::should_fan_out(rows, ctx.cost_class())
    }

    /// Fan the candidate scan across the query pool and merge the partials in
    /// partition order.
    ///
    /// Order is preserved by construction (D5): `par_chunks` partitions by
    /// index range, `collect` on an indexed parallel iterator restores
    /// partition order, and [`ScanPartial::merge_from`] folds them in that
    /// order — so the emitted group order is the sequential scan's first-seen
    /// order, node-for-node.
    fn scan_partitions_parallel<'r>(
        &self,
        nodes: &[NodeIndex],
        ctx: &ScanPartitionCtx<'_, '_>,
        template: &ScanRuntime<'r>,
    ) -> Result<ScanPartial, String>
    where
        'g: 'r,
    {
        #[cfg(test)]
        crate::graph::parallel::PARALLEL_SCANS
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        // Pre-warm (D4): the only structure a scan region touches that is
        // built on first use rather than at load. One call on the calling
        // thread, and every worker then sees a lock-free read.
        let _ = self.property_might_be_alias("");

        let interrupt = ParallelInterrupt::new(|| self.check_deadline().err());
        let partitions = (rayon::current_num_threads() * SCAN_PARTITIONS_PER_WORKER).max(1);
        let chunk_len = nodes.len().div_ceil(partitions).max(1);
        let partials: Vec<ScanPartial> = parallel::install(|| {
            nodes
                .par_chunks(chunk_len)
                .map(|part| {
                    // Each partition forks its own route table; the compiled
                    // trees in `ctx` are read-only and shared.
                    let mut runtime = template.fork();
                    self.scan_partition(part, ctx, &mut runtime, &interrupt)
                })
                .collect::<Result<Vec<_>, String>>()
        })?;
        let mut merged = ScanPartial::default();
        for partial in partials {
            merged.merge_from(partial);
        }
        Ok(merged)
    }
}

// `InlineAccumulators` moved here from `fused_match.rs` when that file
// crossed the 2500-line source-quality ceiling again; `ScanPartial` above
// is its main holder, and the two are always read together.

/// Running aggregate state for one group of a fused node scan.
///
/// Every supported aggregate is folded into the same handful of running values
/// plus, for `count(DISTINCT …)`, a per-aggregate value set — so one pass over
/// the scan answers `count` / `sum` / `avg` / `min` / `max` together, without
/// materialising the group's rows.
///
/// Two counts, not one: `count()` counts non-null values, while `sum()` and
/// `avg()` see only the numeric ones. They diverge on a mixed-type property,
/// which is what made `avg` over `[10, 20, 'hello']` answer 30/3.
struct InlineAccumulators {
    counts: Vec<i64>,
    /// Per-aggregate count of the values that were actually *numeric* — the
    /// divisor `avg()` needs, and the emptiness test `sum()` needs. It differs
    /// from `counts` (all non-null values) exactly when a property holds mixed
    /// types: `[10, 20, 'hello']` is 3 non-null values but 2 numeric ones, and
    /// dividing the numeric sum by 3 is the wrong average.
    numeric_counts: Vec<i64>,
    /// Whether every numeric value this aggregate saw was an `Int64` — the
    /// integer-ness probe `sum()` emits with, matching the streaming
    /// aggregate's `sum_was_int` (`stream::aggregate::AggState`), the path
    /// these queries take when they do not fuse.
    ///
    /// Deriving it from `mins` instead (as this operator used to) reads the
    /// *smallest* value under the cross-type order — where a string outranks
    /// every number — so one string cell turned `sum()` over an integer
    /// property into a float on the fused path only.
    sum_was_int: Vec<bool>,
    sums: Vec<f64>,
    mins: Vec<Option<Value>>,
    maxs: Vec<Option<Value>>,
    /// Per-aggregate value set for `count(DISTINCT …)`; `None` for the rest.
    distinct_sets: Vec<Option<FxHashSet<Value>>>,
}

impl InlineAccumulators {
    fn new(agg_is_distinct: &[bool]) -> Self {
        let width = agg_is_distinct.len();
        InlineAccumulators {
            counts: vec![0i64; width],
            numeric_counts: vec![0i64; width],
            sum_was_int: vec![true; width],
            sums: vec![0.0f64; width],
            mins: vec![None; width],
            maxs: vec![None; width],
            distinct_sets: agg_is_distinct
                .iter()
                .map(|&distinct| distinct.then(FxHashSet::default))
                .collect(),
        }
    }

    /// Fold a later partition's accumulator for the same group into this one.
    ///
    /// The associative combine every aggregate this operator serves admits:
    /// `count`/`sum` add, `min`/`max` take the extreme under the same
    /// `total_order` the row path uses, and `count(DISTINCT …)` unions the
    /// value sets. `avg` is derived from `sums`/`counts` at emission, so it
    /// merges for free. Nothing here reads row order, which is why the
    /// partitioned scan returns byte-identical results.
    fn merge(&mut self, other: InlineAccumulators) {
        for (ai, count) in other.counts.iter().enumerate() {
            self.counts[ai] += count;
            self.numeric_counts[ai] += other.numeric_counts[ai];
            self.sum_was_int[ai] &= other.sum_was_int[ai];
            self.sums[ai] += other.sums[ai];
        }
        for (ai, min) in other.mins.into_iter().enumerate() {
            if let Some(val) = min {
                let replace = match &self.mins[ai] {
                    None => true,
                    Some(cur) => {
                        crate::graph::core::filtering::total_order(&val, cur)
                            == std::cmp::Ordering::Less
                    }
                };
                if replace {
                    self.mins[ai] = Some(val);
                }
            }
        }
        for (ai, max) in other.maxs.into_iter().enumerate() {
            if let Some(val) = max {
                let replace = match &self.maxs[ai] {
                    None => true,
                    Some(cur) => {
                        crate::graph::core::filtering::total_order(&val, cur)
                            == std::cmp::Ordering::Greater
                    }
                };
                if replace {
                    self.maxs[ai] = Some(val);
                }
            }
        }
        for (ai, set) in other.distinct_sets.into_iter().enumerate() {
            if let Some(values) = set {
                self.distinct_sets[ai]
                    .get_or_insert_with(FxHashSet::default)
                    .extend(values);
            }
        }
    }

    /// Fold one row's value for aggregate `ai` into the running state.
    ///
    /// `Null` contributes to nothing: it is not counted, not summed, and never
    /// becomes a `min`/`max` — the same rule the materialized aggregation path
    /// applies. A non-null value that is not numeric (a string in an otherwise
    /// numeric property) counts and can win `min`/`max`, but contributes to
    /// neither `sums` nor `numeric_counts` — `sum`/`avg` see only numbers.
    ///
    /// `count(*)` and `count(<bound var>)` arrive as a non-null `Boolean`
    /// marker, so they count without materialising a node value.
    fn absorb(&mut self, ai: usize, val: &Value, distinct: bool) {
        if distinct {
            // count(DISTINCT …): dedup non-null values in the per-agg set.
            if !matches!(val, Value::Null) {
                self.distinct_sets[ai]
                    .get_or_insert_with(FxHashSet::default)
                    .insert(val.clone());
            }
            return;
        }
        if matches!(val, Value::Null) {
            return;
        }
        self.counts[ai] += 1;
        if let Some(f) = value_to_f64(val) {
            self.numeric_counts[ai] += 1;
            self.sums[ai] += f;
            // UniqueId and Float64 both force a Float64 sum, as they do on the
            // streaming path.
            if !matches!(val, Value::Int64(_)) {
                self.sum_was_int[ai] = false;
            }
        }
        // The is_none() short-circuit guarantees the unwrap can't fire, but
        // the .expect() makes the invariant explicit if a future refactor
        // reorders the conditions.
        if self.mins[ai].is_none()
            || crate::graph::core::filtering::total_order(
                val,
                self.mins[ai]
                    .as_ref()
                    .expect("invariant: is_none() short-circuited above"),
            ) == std::cmp::Ordering::Less
        {
            self.mins[ai] = Some(val.clone());
        }
        if self.maxs[ai].is_none()
            || crate::graph::core::filtering::total_order(
                val,
                self.maxs[ai]
                    .as_ref()
                    .expect("invariant: is_none() short-circuited above"),
            ) == std::cmp::Ordering::Greater
        {
            self.maxs[ai] = Some(val.clone());
        }
    }
}
