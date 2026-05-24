//! Spatial-join executor — handles `Clause::SpatialJoin` produced by
//! `fuse_spatial_join` in the planner.
//!
//! Iterates probe-side nodes once, probes an R-tree built over container
//! bboxes, and emits only matching pairs. Replaces the 2-pattern cartesian
//! + WHERE contains() pipeline with O((N+M) log N + K) work.

use super::super::ast::{Predicate, SpatialProbeKind};
use super::*;
use petgraph::graph::NodeIndex;
use rstar::{RTree, RTreeObject, AABB};
use std::sync::Arc;

/// An R-tree entry: a container node's bbox tagged with its NodeIndex and
/// parsed geometry (Arc-shared with the per-query spatial cache).
struct IndexedContainer {
    node_idx: NodeIndex,
    geom: Arc<geo::Geometry<f64>>,
    min_x: f64,
    min_y: f64,
    max_x: f64,
    max_y: f64,
}

impl RTreeObject for IndexedContainer {
    type Envelope = AABB<[f64; 2]>;

    fn envelope(&self) -> Self::Envelope {
        AABB::from_corners([self.min_x, self.min_y], [self.max_x, self.max_y])
    }
}

impl<'a> CypherExecutor<'a> {
    /// Execute a fused spatial-join clause. Produces one row per (container,
    /// probe) pair where the probe's point lies inside the container's
    /// geometry.
    pub(super) fn execute_spatial_join(
        &self,
        container_var: &str,
        probe_var: &str,
        container_type: &str,
        probe_type: &str,
        probe_kind: SpatialProbeKind,
        remainder: Option<&Predicate>,
    ) -> Result<ResultSet, String> {
        // Fall back to a regular two-pattern MATCH + WHERE if either type has
        // no nodes. This preserves correctness on empty subgraphs.
        let container_indices = match self.graph.type_indices.get(container_type) {
            Some(v) if !v.is_empty() => v,
            _ => return Ok(ResultSet::new()),
        };
        let probe_indices = match self.graph.type_indices.get(probe_type) {
            Some(v) if !v.is_empty() => v,
            _ => return Ok(ResultSet::new()),
        };

        // Build the R-tree entries. Skip containers without parsed geometry
        // or without a computable bbox.
        let mut entries: Vec<IndexedContainer> = Vec::with_capacity(container_indices.len());
        for idx in container_indices.iter() {
            self.ensure_node_spatial_cached(idx);
            let cache = self.spatial_node_cache.read().unwrap();
            if let Some(Some(data)) = cache.get(&idx.index()) {
                if let Some((geom, Some(bbox))) = &data.geometry {
                    entries.push(IndexedContainer {
                        node_idx: idx,
                        geom: Arc::clone(geom),
                        min_x: bbox.min().x,
                        min_y: bbox.min().y,
                        max_x: bbox.max().x,
                        max_y: bbox.max().y,
                    });
                }
            }
        }
        if entries.is_empty() {
            return Ok(ResultSet::new());
        }

        let tree = RTree::<IndexedContainer>::bulk_load(entries);

        let mut rows: Vec<ResultRow> = Vec::new();

        for (probe_i, probe_idx) in probe_indices.iter().enumerate() {
            self.ensure_node_spatial_cached(probe_idx);
            let probe_point: Option<(f64, f64)> = {
                let cache = self.spatial_node_cache.read().unwrap();
                match cache.get(&probe_idx.index()) {
                    Some(Some(data)) => match probe_kind {
                        SpatialProbeKind::Location => data.location,
                        // Probe via the geometry centroid. Drops probes
                        // whose geometry is missing or whose centroid
                        // computation fails (degenerate WKT).
                        SpatialProbeKind::Centroid => {
                            data.geometry.as_ref().and_then(|(geom, _bbox)| {
                                crate::graph::features::spatial::geometry_centroid(geom).ok()
                            })
                        }
                    },
                    _ => None,
                }
            };
            let (lat, lon) = match probe_point {
                Some(pt) => pt,
                None => continue,
            };

            // R-tree bbox probe: only containers whose bbox contains this point.
            let env = AABB::from_point([lon, lat]);
            for cand in tree.locate_in_envelope_intersecting(&env) {
                let pt = geo::Point::new(lon, lat);
                if crate::graph::features::spatial::geometry_contains_point(&cand.geom, &pt) {
                    let mut row = ResultRow::with_capacity(2, 0, 0);
                    row.node_bindings
                        .insert(container_var.to_string(), cand.node_idx);
                    row.node_bindings.insert(probe_var.to_string(), probe_idx);
                    rows.push(row);
                }
            }

            if probe_i & 2047 == 0 {
                self.check_deadline()?;
            }
        }

        // Apply the residual WHERE predicate (e.g. the AND-remainder after
        // `contains()` was stripped during planning).
        if let Some(pred) = remainder {
            let mut keep = Vec::with_capacity(rows.len());
            for row in rows {
                match self.evaluate_predicate(pred, &row) {
                    Ok(true) => keep.push(row),
                    Ok(false) => {}
                    Err(e) => return Err(e),
                }
            }
            rows = keep;
        }

        Ok(ResultSet {
            rows,
            columns: Vec::new(),
            lazy_return_items: None,
        })
    }
}
