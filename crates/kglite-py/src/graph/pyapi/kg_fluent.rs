//! KnowledgeGraph #[pymethods]: selection / filter / traversal chain.
//!
//! Part of the Phase 9 split of the kg_methods.rs monolith (5,419 lines
//! single pymethods block). PyO3 merges multiple `#[pymethods] impl`
//! blocks at class-registration time, so the split is purely structural —
//! no runtime impact.

use crate::datatypes::values::{FilterCondition, Value};
use crate::datatypes::{py_in, py_out};
use crate::graph::{get_graph_mut, KnowledgeGraph, TemporalContext};
use kglite_core::api::GraphRead;
use kglite_core::api::OperationReport;
use kglite_core::api::{CowSelection, PlanStep};
use petgraph::graph::NodeIndex;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3::Bound;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// Map a core `filtering::*` error string into the fluent-API `PyErr`. Shared
/// by every selection-mutator routed through `derive_with`.
fn fluent_arg_err(e: String) -> PyErr {
    crate::error_py::kg_to_pyerr(crate::error::KgError::Argument(e))
}

#[pymethods]
impl KnowledgeGraph {
    /// Configure temporal validity for a node type or connection type.
    ///
    /// After configuration, `select()` auto-filters temporal nodes to "current" and
    /// `traverse()` auto-filters temporal connections to "current". Use `date()` to
    /// shift the temporal context.
    ///
    /// Args:
    ///     type_name: Node type (e.g. "FieldStatus") or connection type (e.g. "HAS_LICENSEE").
    ///     valid_from: Property name holding the start date (e.g. "fldLicenseeFrom").
    ///     valid_to: Property name holding the end date (e.g. "fldLicenseeTo").
    #[pyo3(signature = (type_name, valid_from, valid_to))]
    fn set_temporal(
        &mut self,
        type_name: String,
        valid_from: String,
        valid_to: String,
    ) -> PyResult<()> {
        use kglite_core::api::TemporalConfig;
        let config = TemporalConfig {
            valid_from,
            valid_to,
        };
        let graph = get_graph_mut(&mut self.inner);
        // Auto-detect: check node types first, then connection types
        if graph.type_indices.contains_key(&type_name) {
            graph.temporal_node_configs.insert(type_name, config);
        } else if graph.connection_type_metadata.contains_key(&type_name) {
            graph
                .temporal_edge_configs
                .entry(type_name)
                .or_default()
                .push(config);
        } else {
            return Err(crate::error_py::kg_to_pyerr(
                crate::error::KgError::Argument(format!(
                    "'{}' is not a known node type or connection type",
                    type_name
                )),
            ));
        }
        Ok(())
    }

    /// Set the temporal context for auto-filtering.
    ///
    /// Returns a new KnowledgeGraph. All subsequent `select()` and `traverse()`
    /// calls on the returned graph use this context for temporal filtering.
    ///
    /// - `date("2013")` — point-in-time (Jan 1 2013)
    /// - `date("2010", "2015")` — range: include anything valid during 2010-2015
    /// - `date("all")` — disable temporal filtering entirely
    /// - `date()` — reset to today
    #[pyo3(signature = (date_str=None, end_str=None))]
    fn date(&self, date_str: Option<&str>, end_str: Option<&str>) -> PyResult<Self> {
        let mut new_kg = self.clone();
        new_kg.cursor.temporal_context = match (date_str, end_str) {
            (Some("all"), _) => TemporalContext::All,
            (Some(start), Some(end)) => {
                let (start_date, _) = kglite_core::api::timeseries::parse_date_query(start)
                    .map_err(|e: String| -> PyErr {
                        crate::error_py::kg_to_pyerr(crate::error::KgError::Argument(e))
                    })?;
                let (end_date, end_precision) = kglite_core::api::timeseries::parse_date_query(end)
                    .map_err(|e: String| -> PyErr {
                        crate::error_py::kg_to_pyerr(crate::error::KgError::Argument(e))
                    })?;
                let expanded_end =
                    kglite_core::api::timeseries::expand_end(end_date, end_precision);
                TemporalContext::During(start_date, expanded_end)
            }
            (Some(s), None) => {
                let (date, _) = kglite_core::api::timeseries::parse_date_query(s).map_err(
                    |e: String| -> PyErr {
                        crate::error_py::kg_to_pyerr(crate::error::KgError::Argument(e))
                    },
                )?;
                TemporalContext::At(date)
            }
            (None, None) => TemporalContext::Today,
            (None, Some(_)) => {
                return Err(crate::error_py::kg_to_pyerr(
                    crate::error::KgError::Argument(
                        "date() end_str requires a start date_str".to_string(),
                    ),
                ));
            }
        };
        Ok(new_kg)
    }

    #[pyo3(signature = (node_type, sort=None, limit=None, temporal=None, include_secondary=false))]
    fn select(
        &mut self,
        node_type: String,
        sort: Option<&Bound<'_, PyAny>>,
        limit: Option<usize>,
        temporal: Option<bool>,
        include_secondary: bool,
    ) -> PyResult<Self> {
        let mut new_kg = self.clone();

        // Record plan step: estimate based on the candidate set. When
        // include_secondary, the count is primary ∪ secondary.
        let estimated = if include_secondary {
            self.inner.nodes_with_label(&node_type).len()
        } else {
            self.inner
                .type_indices
                .get(&node_type)
                .map(|v| v.len())
                .unwrap_or(0)
        };
        new_kg.cursor.selection.clear_execution_plan(); // Start fresh plan

        let sort_fields = if let Some(spec) = sort {
            match spec.extract::<String>() {
                Ok(field) => Some(vec![(field, true)]),
                Err(_) => Some(py_in::parse_sort_fields(spec, None)?),
            }
        } else {
            None
        };

        let to_pyerr = |e: String| -> PyErr {
            crate::error_py::kg_to_pyerr(crate::error::KgError::Argument(e))
        };

        if include_secondary {
            // Seed from nodes carrying `node_type` as primary OR secondary
            // label. On a single-label graph this is identical to the
            // primary `type == node_type` filter below.
            kglite_core::api::fluent::filter_nodes_by_label(
                &self.inner,
                &mut new_kg.cursor.selection,
                &node_type,
                sort_fields,
                limit,
            )
            .map_err(to_pyerr)?;
        } else {
            let mut conditions = HashMap::new();
            conditions.insert(
                "type".to_string(),
                FilterCondition::Equals(Value::String(node_type.clone())),
            );
            kglite_core::api::fluent::filter_nodes(
                &self.inner,
                &mut new_kg.cursor.selection,
                conditions,
                sort_fields,
                limit,
            )
            .map_err(to_pyerr)?;
        }

        // Apply temporal filtering if configured and not disabled
        if temporal != Some(false) && !self.cursor.temporal_context.is_all() {
            if let Some(config) = self.inner.temporal_node_configs.get(&node_type) {
                let level_idx = new_kg.cursor.selection.get_level_count().saturating_sub(1);
                if let Some(level) = new_kg.cursor.selection.get_level_mut(level_idx) {
                    for nodes in level.selections.values_mut() {
                        nodes.retain(|&idx| {
                            if let Some(node) = self.inner.graph.node_weight(idx) {
                                kglite_core::api::fluent::node_passes_context(
                                    node,
                                    config,
                                    &self.cursor.temporal_context,
                                )
                            } else {
                                false
                            }
                        });
                    }
                }
            }
        }

        // Record actual result
        let actual = new_kg
            .cursor
            .selection
            .get_level(new_kg.cursor.selection.get_level_count().saturating_sub(1))
            .map(|l| l.node_count())
            .unwrap_or(0);
        new_kg.cursor.selection.add_plan_step(
            PlanStep::new("SELECT", Some(&node_type), estimated).with_actual_rows(actual),
        );

        Ok(new_kg)
    }

    #[pyo3(signature = (conditions, sort=None, limit=None))]
    #[pyo3(name = "where")]
    fn where_method(
        &self,
        conditions: &Bound<'_, PyDict>,
        sort: Option<&Bound<'_, PyAny>>,
        limit: Option<usize>,
    ) -> PyResult<Self> {
        let filter_conditions = py_in::pydict_to_filter_conditions(conditions)?;
        let sort_fields = match sort {
            Some(spec) => Some(py_in::parse_sort_fields(spec, None)?),
            None => None,
        };

        self.derive_with(|inner, cursor| {
            let level_count = |c: &crate::graph::CursorState| {
                c.selection
                    .get_level(c.selection.get_level_count().saturating_sub(1))
                    .map(|l| l.node_count())
                    .unwrap_or(0)
            };
            // Estimate based on the (just-cloned) current selection.
            let estimated = level_count(cursor);
            kglite_core::api::fluent::filter_nodes(
                inner,
                &mut cursor.selection,
                filter_conditions,
                sort_fields,
                limit,
            )
            .map_err(fluent_arg_err)?;
            let actual = level_count(cursor);
            cursor
                .selection
                .add_plan_step(PlanStep::new("WHERE", None, estimated).with_actual_rows(actual));
            Ok(())
        })
    }

    /// Filter nodes matching ANY of the given condition sets (OR logic).
    /// Each item in the list is a condition dict (same format as where()).
    /// A node is kept if it matches at least one condition set.
    #[pyo3(signature = (conditions, sort=None, limit=None))]
    fn where_any(
        &self,
        conditions: &Bound<'_, PyList>,
        sort: Option<&Bound<'_, PyAny>>,
        limit: Option<usize>,
    ) -> PyResult<Self> {
        let condition_sets: Vec<HashMap<String, FilterCondition>> = conditions
            .iter()
            .map(|item| {
                let dict = item.cast::<PyDict>().map_err(|_| -> PyErr {
                    crate::error_py::kg_to_pyerr(crate::error::KgError::Argument(
                        "where_any expects a list of condition dicts".to_string(),
                    ))
                })?;
                py_in::pydict_to_filter_conditions(dict)
            })
            .collect::<PyResult<Vec<_>>>()?;

        if condition_sets.is_empty() {
            return Err(crate::error_py::kg_to_pyerr(
                crate::error::KgError::Argument(
                    "where_any requires at least one condition set".to_string(),
                ),
            ));
        }

        let sort_fields = match sort {
            Some(spec) => Some(py_in::parse_sort_fields(spec, None)?),
            None => None,
        };

        self.derive_with(|inner, cursor| {
            kglite_core::api::fluent::filter_nodes_any(
                inner,
                &mut cursor.selection,
                &condition_sets,
                sort_fields,
                limit,
            )
            .map_err(fluent_arg_err)
        })
    }

    #[pyo3(signature = (include_orphans=None, sort=None, limit=None))]
    fn where_orphans(
        &self,
        include_orphans: Option<bool>,
        sort: Option<&Bound<'_, PyAny>>,
        limit: Option<usize>,
    ) -> PyResult<Self> {
        let include = include_orphans.unwrap_or(true);
        let sort_fields = if let Some(spec) = sort {
            Some(py_in::parse_sort_fields(spec, None)?)
        } else {
            None
        };

        self.derive_with(|inner, cursor| {
            kglite_core::api::fluent::filter_orphan_nodes(
                inner,
                &mut cursor.selection,
                include,
                sort_fields.as_ref(),
                limit,
            )
            .map_err(fluent_arg_err)
        })
    }

    #[pyo3(signature = (sort, ascending=None))]
    fn sort(&self, sort: &Bound<'_, PyAny>, ascending: Option<bool>) -> PyResult<Self> {
        let sort_fields = py_in::parse_sort_fields(sort, ascending)?;
        self.derive_with(|inner, cursor| {
            kglite_core::api::fluent::sort_nodes(inner, &mut cursor.selection, sort_fields)
                .map_err(fluent_arg_err)
        })
    }

    fn limit(&self, max_per_group: usize) -> PyResult<Self> {
        self.derive_with(|inner, cursor| {
            kglite_core::api::fluent::limit_nodes_per_group(
                inner,
                &mut cursor.selection,
                max_per_group,
            )
            .map_err(fluent_arg_err)
        })
    }

    /// Skip the first N nodes per group (for pagination).
    /// Use with sort() + limit() for paged results:
    ///   graph.sort('name').offset(20).limit(10)
    fn offset(&self, n: usize) -> PyResult<Self> {
        self.derive_with(|inner, cursor| {
            kglite_core::api::fluent::offset_nodes(inner, &mut cursor.selection, n)
                .map_err(fluent_arg_err)
        })
    }

    /// Filter current selection to nodes that have at least one connection
    /// of the given type. Equivalent to Cypher's WHERE EXISTS {(n)-[:TYPE]->()}.
    #[pyo3(signature = (connection_type, direction=None))]
    fn where_connected(&self, connection_type: &str, direction: Option<&str>) -> PyResult<Self> {
        let dir = match direction.unwrap_or("any") {
            "outgoing" | "out" => Some(petgraph::Direction::Outgoing),
            "incoming" | "in" => Some(petgraph::Direction::Incoming),
            "any" | "both" => None,
            d => {
                return Err(crate::error_py::kg_to_pyerr(
                    crate::error::KgError::Argument(format!(
                        "Invalid direction '{}'. Use 'outgoing', 'incoming', or 'any'",
                        d
                    )),
                ))
            }
        };

        self.derive_with(|inner, cursor| {
            kglite_core::api::fluent::filter_by_connection(
                inner,
                &mut cursor.selection,
                connection_type,
                dir,
            )
            .map_err(fluent_arg_err)
        })
    }

    /// Filter nodes that are valid at a specific date
    ///
    /// This is a convenience method for temporal queries. It filters nodes where:
    /// - date_from_field <= date <= date_to_field
    ///
    /// If field names are not specified, auto-detects from set_temporal() config.
    /// If date is not specified, uses the reference date from date() or today.
    #[pyo3(signature = (date=None, date_from_field=None, date_to_field=None))]
    fn valid_at(
        &mut self,
        date: Option<&str>,
        date_from_field: Option<&str>,
        date_to_field: Option<&str>,
    ) -> PyResult<Self> {
        // Auto-detect field names from temporal config if not provided
        let temporal_config = if date_from_field.is_none() || date_to_field.is_none() {
            self.infer_selection_node_type()
                .and_then(|nt| self.inner.temporal_node_configs.get(&nt).cloned())
        } else {
            None
        };
        let from_field = date_from_field
            .map(|s| s.to_string())
            .or_else(|| temporal_config.as_ref().map(|c| c.valid_from.clone()))
            .unwrap_or_else(|| "date_from".to_string());
        let to_field = date_to_field
            .map(|s| s.to_string())
            .or_else(|| temporal_config.as_ref().map(|c| c.valid_to.clone()))
            .unwrap_or_else(|| "date_to".to_string());
        // Resolve the reference date
        let ref_date = match date {
            Some(d) => {
                let (parsed, _) = kglite_core::api::timeseries::parse_date_query(d).map_err(
                    |e: String| -> PyErr {
                        crate::error_py::kg_to_pyerr(crate::error::KgError::Argument(e))
                    },
                )?;
                parsed
            }
            None => match &self.cursor.temporal_context {
                TemporalContext::At(d) => *d,
                _ => chrono::Local::now().date_naive(),
            },
        };

        // Use temporal helper for NULL-aware filtering (NULL date_to = still active)
        let config = kglite_core::api::TemporalConfig {
            valid_from: from_field,
            valid_to: to_field,
        };

        let mut new_kg = self.clone();

        // Estimate based on current selection
        let estimated = new_kg
            .cursor
            .selection
            .get_level(new_kg.cursor.selection.get_level_count().saturating_sub(1))
            .map(|l| l.node_count())
            .unwrap_or(0);

        // Filter in-place using temporal validity (handles NULL as unbounded)
        let current_level = new_kg.cursor.selection.get_level_count().saturating_sub(1);
        if let Some(level) = new_kg.cursor.selection.get_level_mut(current_level) {
            for (_parent, children) in level.selections.iter_mut() {
                children.retain(|&idx| {
                    if let Some(node) = self.inner.graph.node_weight(idx) {
                        kglite_core::api::fluent::node_is_temporally_valid(node, &config, &ref_date)
                    } else {
                        false
                    }
                });
            }
        }

        // Record actual result
        let actual = new_kg
            .cursor
            .selection
            .get_level(new_kg.cursor.selection.get_level_count().saturating_sub(1))
            .map(|l| l.node_count())
            .unwrap_or(0);
        new_kg
            .cursor
            .selection
            .add_plan_step(PlanStep::new("VALID_AT", None, estimated).with_actual_rows(actual));

        Ok(new_kg)
    }

    /// Filter nodes that are valid during a date range
    ///
    /// This filters nodes where their validity period overlaps with the given range:
    /// - date_from_field <= end_date AND date_to_field >= start_date
    ///
    /// If field names are not specified, auto-detects from set_temporal() config.
    #[pyo3(signature = (start_date, end_date, date_from_field=None, date_to_field=None))]
    fn valid_during(
        &mut self,
        start_date: &str,
        end_date: &str,
        date_from_field: Option<&str>,
        date_to_field: Option<&str>,
    ) -> PyResult<Self> {
        // Auto-detect field names from temporal config if not provided
        let temporal_config = if date_from_field.is_none() || date_to_field.is_none() {
            self.infer_selection_node_type()
                .and_then(|nt| self.inner.temporal_node_configs.get(&nt).cloned())
        } else {
            None
        };
        let from_field = date_from_field
            .map(|s| s.to_string())
            .or_else(|| temporal_config.as_ref().map(|c| c.valid_from.clone()))
            .unwrap_or_else(|| "date_from".to_string());
        let to_field = date_to_field
            .map(|s| s.to_string())
            .or_else(|| temporal_config.as_ref().map(|c| c.valid_to.clone()))
            .unwrap_or_else(|| "date_to".to_string());

        // Parse dates
        let (start_parsed, _) = kglite_core::api::timeseries::parse_date_query(start_date)
            .map_err(|e: String| -> PyErr {
                crate::error_py::kg_to_pyerr(crate::error::KgError::Argument(e))
            })?;
        let (end_parsed, _) = kglite_core::api::timeseries::parse_date_query(end_date).map_err(
            |e: String| -> PyErr {
                crate::error_py::kg_to_pyerr(crate::error::KgError::Argument(e))
            },
        )?;

        // Use temporal helper for NULL-aware overlap check
        let config = kglite_core::api::TemporalConfig {
            valid_from: from_field,
            valid_to: to_field,
        };

        let mut new_kg = self.clone();

        // Estimate based on current selection
        let estimated = new_kg
            .cursor
            .selection
            .get_level(new_kg.cursor.selection.get_level_count().saturating_sub(1))
            .map(|l| l.node_count())
            .unwrap_or(0);

        // Filter in-place using temporal overlap (handles NULL as unbounded)
        let current_level = new_kg.cursor.selection.get_level_count().saturating_sub(1);
        if let Some(level) = new_kg.cursor.selection.get_level_mut(current_level) {
            for (_parent, children) in level.selections.iter_mut() {
                children.retain(|&idx| {
                    if let Some(node) = self.inner.graph.node_weight(idx) {
                        kglite_core::api::fluent::node_overlaps_range(
                            node,
                            &config,
                            &start_parsed,
                            &end_parsed,
                        )
                    } else {
                        false
                    }
                });
            }
        }

        // Record actual result
        let actual = new_kg
            .cursor
            .selection
            .get_level(new_kg.cursor.selection.get_level_count().saturating_sub(1))
            .map(|l| l.node_count())
            .unwrap_or(0);
        new_kg
            .cursor
            .selection
            .add_plan_step(PlanStep::new("VALID_DURING", None, estimated).with_actual_rows(actual));

        Ok(new_kg)
    }

    /// Update properties on all currently selected nodes
    ///
    /// This allows batch updating of properties on nodes matching the current selection.
    /// Returns a dictionary containing:
    ///   - 'graph': A new KnowledgeGraph with the updated nodes (original is unchanged)
    ///   - 'nodes_updated': Number of nodes that were updated
    ///   - 'report_index': Index of the operation report
    ///
    /// Example:
    ///     ```python
    ///     result = graph.select('Discovery').where({'year': {'>=': 2020}}).update({
    ///         'is_recent': True
    ///     })
    ///     graph = result['graph']  # Use the returned graph with updates
    ///     print(f"Updated {result['nodes_updated']} nodes")
    ///     ```
    ///
    /// Args:
    ///     properties: Dictionary of property names and values to set
    ///     keep_selection: If True, preserve the current selection in the returned graph
    ///
    /// Returns:
    ///     Dictionary with 'graph' (KnowledgeGraph), 'nodes_updated' (int), 'report_index' (int)
    #[pyo3(signature = (properties, keep_selection=None))]
    fn update(
        &mut self,
        properties: &Bound<'_, PyDict>,
        keep_selection: Option<bool>,
    ) -> PyResult<Py<PyAny>> {
        // Get the current level's nodes
        let current_index = self.cursor.selection.get_level_count().saturating_sub(1);
        let level = self
            .cursor
            .selection
            .get_level(current_index)
            .ok_or_else(|| -> PyErr {
                crate::error_py::kg_to_pyerr(crate::error::KgError::Argument(
                    "No active selection level".to_string(),
                ))
            })?;

        let nodes = level.get_all_nodes();
        if nodes.is_empty() {
            return Err(crate::error_py::kg_to_pyerr(
                crate::error::KgError::Argument("No nodes selected for update".to_string()),
            ));
        }

        // Pre-extract Python values before mutating the graph
        let mut parsed_properties: Vec<(String, Value)> = Vec::new();
        for (key, value) in properties.iter() {
            let property_name: String = key.extract().map_err(|_| {
                crate::error_py::kg_to_pyerr(crate::error::KgError::Argument(
                    "Property names must be strings".to_string(),
                ))
            })?;
            let property_value = py_in::py_value_to_value(&value)?;
            parsed_properties.push((property_name, property_value));
        }

        // Now mutate the graph — no ? operators from here to Arc creation
        let graph = get_graph_mut(&mut self.inner);

        let mut total_updated = 0;
        let mut errors = Vec::new();

        for (property_name, property_value) in &parsed_properties {
            let node_values: Vec<(Option<petgraph::graph::NodeIndex>, Value)> = nodes
                .iter()
                .map(|&idx| (Some(idx), property_value.clone()))
                .collect();

            match kglite_core::api::mutation::update_node_properties(
                graph,
                &node_values,
                property_name,
            ) {
                Ok(report) => {
                    total_updated += report.nodes_updated;
                    errors.extend(report.errors);
                }
                Err(e) => {
                    errors.push(format!(
                        "Error updating property '{}': {}",
                        property_name, e
                    ));
                }
            }
        }

        // Create the result KnowledgeGraph (clone the Arc for the new graph)
        let mut new_kg = KnowledgeGraph {
            inner: self.inner.clone(),
            cursor: crate::graph::CursorState {
                selection: if keep_selection.unwrap_or(false) {
                    self.cursor.selection.clone()
                } else {
                    CowSelection::new()
                },
                reports: self.cursor.reports.clone(),
                last_mutation_stats: None,
                temporal_context: self.cursor.temporal_context.clone(),
            },
            embedder: self.embedder.as_ref().map(Arc::clone),
            default_timeout_ms: self.default_timeout_ms,
            default_max_rows: self.default_max_rows,
            lifecycle: crate::graph::GraphLifecycle::detached(),
        };

        // Create and add a report
        let report = kglite_core::api::NodeOperationReport {
            operation_type: "update".to_string(),
            timestamp: chrono::Utc::now(),
            nodes_created: 0,
            nodes_updated: total_updated,
            nodes_skipped: 0,
            processing_time_ms: 0.0, // Could track this if needed
            errors,
        };

        let report_index = new_kg.add_report(OperationReport::NodeOperation(report));

        // Return the new KnowledgeGraph and the report
        Python::attach(|py| {
            let dict = PyDict::new(py);
            dict.set_item("graph", Py::new(py, new_kg)?.into_any())?;
            dict.set_item("nodes_updated", total_updated)?;
            dict.set_item("report_index", report_index)?;
            Ok(dict.into())
        })
    }

    /// Materialise selected nodes as a flat ``ResultView``.
    #[pyo3(signature = (limit=None))]
    fn collect(&self, limit: Option<usize>) -> PyResult<Py<PyAny>> {
        let max = limit.unwrap_or(usize::MAX);
        let node_indices: Vec<petgraph::graph::NodeIndex> = self
            .cursor
            .selection
            .current_node_indices()
            .take(max)
            .collect();
        let view = crate::graph::pyapi::result_view::ResultView::from_nodes_with_graph(
            &self.inner,
            &node_indices,
        );
        Python::attach(|py| Py::new(py, view).map(|v| v.into_any()))
    }

    /// Materialise selected nodes grouped by a parent type in the traversal
    /// hierarchy. Always returns a ``dict``.
    #[pyo3(signature = (group_by, *, parent_info=false, flatten_single_parent=true, limit=None))]
    fn collect_grouped(
        &self,
        group_by: &str,
        parent_info: Option<bool>,
        flatten_single_parent: Option<bool>,
        limit: Option<usize>,
    ) -> PyResult<Py<PyAny>> {
        let nodes = kglite_core::api::fluent::get_nodes(
            &self.inner,
            &self.cursor.selection,
            None,
            None,
            limit,
        );
        Python::attach(|py| {
            py_out::level_nodes_to_pydict(
                py,
                &nodes,
                Some(group_by),
                parent_info,
                flatten_single_parent,
            )
        })
    }

    /// Export the current selection as a pandas DataFrame.
    ///
    /// Each node becomes a row with columns for title, type, id, and all properties.
    /// Nodes of different types may have different properties — missing values become None.
    #[pyo3(signature = (*, include_type=true, include_id=true))]
    fn to_df(&self, py: Python<'_>, include_type: bool, include_id: bool) -> PyResult<Py<PyAny>> {
        // Collect nodes from the current selection
        let mut nodes_data: Vec<(&str, &kglite_core::api::NodeData)> = Vec::new();
        for node_idx in self.cursor.selection.current_node_indices() {
            if let Some(node) = self.inner.get_node(node_idx) {
                nodes_data.push((node.node_type_str(&self.inner.interner), node));
            }
        }

        // Fast path: use TypeSchema for key discovery when all nodes share a type
        let prop_keys: Vec<String> = if nodes_data.len() > 50 {
            let first_type = nodes_data[0].0;
            let all_same = nodes_data.iter().all(|(nt, _)| *nt == first_type);
            if all_same {
                if let Some(schema) = self.inner.type_schemas.get(first_type) {
                    let mut keys: Vec<String> = schema
                        .iter()
                        .filter_map(|(_, ik)| {
                            self.inner.interner.try_resolve(ik).map(|s| s.to_string())
                        })
                        .collect();
                    keys.sort();
                    keys
                } else {
                    Self::discover_property_keys_from_data(&nodes_data, &self.inner.interner)
                }
            } else {
                Self::discover_property_keys_from_data(&nodes_data, &self.inner.interner)
            }
        } else {
            Self::discover_property_keys_from_data(&nodes_data, &self.inner.interner)
        };

        // Build columnar dict-of-lists
        let n = nodes_data.len();
        let title_col = PyList::empty(py);
        let type_col = if include_type {
            Some(PyList::empty(py))
        } else {
            None
        };
        let id_col = if include_id {
            Some(PyList::empty(py))
        } else {
            None
        };

        // Pre-create property column lists
        let prop_cols: Vec<pyo3::Bound<'_, PyList>> =
            prop_keys.iter().map(|_| PyList::empty(py)).collect();

        for (node_type, node) in &nodes_data {
            title_col.append(py_out::value_to_py(py, &node.title())?)?;
            if let Some(ref tc) = type_col {
                tc.append(*node_type)?;
            }
            if let Some(ref ic) = id_col {
                ic.append(py_out::value_to_py(py, &node.id())?)?;
            }
            for (j, key) in prop_keys.iter().enumerate() {
                let val = node.get_property(key);
                let val_ref = val.as_deref().unwrap_or(&Value::Null);
                prop_cols[j].append(py_out::value_to_py(py, val_ref)?)?;
            }
        }

        // Build the dict with ordered columns: type, title, id, ...properties
        let dict = PyDict::new(py);
        let columns = PyList::empty(py);

        if let Some(tc) = type_col {
            dict.set_item("type", tc)?;
            columns.append("type")?;
        }
        dict.set_item("title", title_col)?;
        columns.append("title")?;
        if let Some(ic) = id_col {
            dict.set_item("id", ic)?;
            columns.append("id")?;
        }
        for (j, key) in prop_keys.iter().enumerate() {
            dict.set_item(key, &prop_cols[j])?;
            columns.append(key)?;
        }

        let pd = py.import("pandas")?;

        if n == 0 {
            return pd.call_method0("DataFrame").map(|df| df.unbind());
        }

        // Create DataFrame with column order preserved
        let kwargs = PyDict::new(py);
        kwargs.set_item("columns", columns)?;
        let df = pd.call_method("DataFrame", (dict,), Some(&kwargs))?;
        Ok(df.unbind())
    }

    /// Format the current selection as a human-readable string.
    ///
    /// Each node is printed as a block with type, id, title, and all properties.
    /// The ``limit`` parameter caps the number of nodes shown (default 50).
    #[pyo3(signature = (limit=50))]
    fn to_str(&self, limit: usize) -> PyResult<String> {
        use crate::datatypes::values::format_value;

        let node_indices: Vec<_> = self.cursor.selection.current_node_indices().collect();
        let total = node_indices.len();
        let show = total.min(limit);

        if total == 0 {
            return Ok("(empty selection)".to_string());
        }

        let mut buf = String::with_capacity(show * 200);

        for (i, &idx) in node_indices.iter().take(show).enumerate() {
            if let Some(node) = self.inner.get_node(idx) {
                if i > 0 {
                    buf.push('\n');
                }
                buf.push_str(&format!(
                    "[{}] {} (id: {})\n",
                    node.node_type_str(&self.inner.interner),
                    format_value(&node.title()),
                    format_value(&node.id()),
                ));
                // Sort property keys for deterministic output
                let mut keys: Vec<&str> = node.property_keys(&self.inner.interner).collect();
                keys.sort();
                for key in keys {
                    if let Some(val) = node.get_property(key) {
                        let s = format_value(&val);
                        let display = if s.len() > 80 {
                            let keep = (80 - 5) / 2;
                            format!("{} ... {}", &s[..keep], &s[s.len() - keep..])
                        } else {
                            s
                        };
                        buf.push_str(&format!("  {}: {}\n", key, display));
                    }
                }
            }
        }

        if total > show {
            buf.push_str(&format!("\n... and {} more nodes\n", total - show));
        }

        Ok(buf)
    }

    /// Display selected nodes with specific properties in a compact format.
    ///
    /// Single level (no traversals): one node per line as `Type(val1, val2)`
    /// Multi-level (after traverse): walks the full chain as
    /// `Type1(vals) -> Type2(vals) -> Type3(vals)`
    ///
    /// Args:
    ///     columns: property names to include (default: ["id", "title"])
    ///     limit: max output lines (default: 200)
    ///
    /// Example:
    ///     ```python
    ///     print(graph.select("Discovery").show(["id", "title"]))
    ///     # Discovery(123, Johan Sverdrup)
    ///     # Discovery(456, Troll)
    ///
    /// ```text
    /// print(graph.select("Discovery")
    ///     .traverse("HAS_DEPOSIT_PROSPECT")
    ///     .traverse("TESTED_BY_WELLBORE")
    ///     .show(["id", "title"]))
    /// # Discovery(123, Johan Sverdrup) -> Prospect(456, Alpha) -> Wellbore(789, W1)
    /// ```
    /// ```
    #[pyo3(signature = (columns=None, limit=200))]
    fn show(&self, columns: Option<Vec<String>>, limit: usize) -> PyResult<String> {
        use kglite_core::api::fluent::format_value_compact;

        let columns = columns.unwrap_or_else(|| vec!["id".to_string(), "title".to_string()]);
        let level_count = self.cursor.selection.get_level_count();

        // Helper: format a single node as Type(val1, val2, ...)
        let fmt_node = |idx: NodeIndex| -> String {
            let node = match self.inner.get_node(idx) {
                Some(n) => n,
                None => return "?".to_string(),
            };
            let mut s = String::with_capacity(64);
            let node_type_str = node.node_type_str(&self.inner.interner);
            s.push_str(node_type_str);
            s.push('(');
            let mut first = true;
            for col in &columns {
                let resolved = self.inner.resolve_alias(node_type_str, col);
                if let Some(val) = node.get_field_ref(resolved) {
                    if matches!(&*val, Value::Null) {
                        continue;
                    }
                    if !first {
                        s.push_str(", ");
                    }
                    let v = format_value_compact(&val);
                    if v.len() > 80 {
                        let keep = (80 - 5) / 2;
                        s.push_str(&v[..keep]);
                        s.push_str(" ... ");
                        s.push_str(&v[v.len() - keep..]);
                    } else {
                        s.push_str(&v);
                    }
                    first = false;
                }
            }
            s.push(')');
            s
        };

        if level_count <= 1 {
            // Single level: format each node on its own line
            let nodes: Vec<_> = self.cursor.selection.current_node_indices().collect();
            if nodes.is_empty() {
                return Ok("(empty selection)".to_string());
            }
            let show_count = nodes.len().min(limit);
            let mut buf = String::with_capacity(show_count * 80);
            for &idx in nodes.iter().take(show_count) {
                buf.push_str(&fmt_node(idx));
                buf.push('\n');
            }
            if nodes.len() > show_count {
                buf.push_str(&format!("... and {} more\n", nodes.len() - show_count));
            }
            Ok(buf)
        } else {
            // Multi-level: walk traversal chains via DFS
            let level0 = self.cursor.selection.get_level(0).ok_or_else(|| {
                crate::error_py::kg_to_pyerr(crate::error::KgError::CypherExecution {
                    message: ("no selection levels").to_string(),
                    position: None,
                })
            })?;

            let mut chains: Vec<Vec<NodeIndex>> = Vec::new();
            let roots = level0.get_all_nodes();

            'outer: for root in &roots {
                let mut stack: Vec<(usize, Vec<NodeIndex>)> = vec![(1, vec![*root])];

                while let Some((level_idx, chain)) = stack.pop() {
                    if chains.len() >= limit {
                        break 'outer;
                    }

                    if level_idx >= level_count {
                        // Reached the end — complete chain
                        chains.push(chain);
                        continue;
                    }

                    let level = match self.cursor.selection.get_level(level_idx) {
                        Some(l) => l,
                        None => {
                            chains.push(chain);
                            continue;
                        }
                    };

                    let last_node = *chain.last().unwrap();
                    match level.selections.get(&Some(last_node)) {
                        Some(children) if !children.is_empty() => {
                            for &child in children {
                                let mut new_chain = chain.clone();
                                new_chain.push(child);
                                stack.push((level_idx + 1, new_chain));
                            }
                        }
                        _ => {
                            // Dead end — omit incomplete chains
                        }
                    }
                }
            }

            if chains.is_empty() {
                return Ok("(no traversal results)".to_string());
            }

            let show_count = chains.len().min(limit);
            let mut buf = String::with_capacity(show_count * 120);
            for chain in chains.iter().take(show_count) {
                for (i, &idx) in chain.iter().enumerate() {
                    if i > 0 {
                        buf.push_str(" -> ");
                    }
                    buf.push_str(&fmt_node(idx));
                }
                buf.push('\n');
            }
            if chains.len() > show_count {
                buf.push_str(&format!(
                    "... and {} more chains\n",
                    chains.len() - show_count
                ));
            }
            Ok(buf)
        }
    }

    /// Returns the count of nodes in the current selection without materialization.
    /// If no selection has been applied, returns the total graph node count.
    /// Much faster than collect() when you only need the count.
    /// Also available via Python's built-in len(): len(graph.select('User'))
    ///
    /// Example:
    ///     ```python
    ///     count = graph.len()                      # total nodes in graph
    ///     count = graph.select('User').len()        # filtered count
    ///     count = len(graph.select('User'))         # same, via __len__
    ///     ```
    #[pyo3(name = "len")]
    fn py_len(&self) -> usize {
        if self.cursor.selection.has_active_selection() {
            self.cursor.selection.current_node_count()
        } else {
            self.inner.graph.node_count()
        }
    }

    fn __len__(&self) -> usize {
        self.py_len()
    }

    /// Returns the raw node indices in the current selection.
    /// Much faster than collect() when you only need indices for further processing.
    ///
    /// Example:
    ///     ```python
    ///     indices = graph.select('User').indices()
    ///     ```
    fn indices(&self) -> Vec<usize> {
        self.cursor
            .selection
            .current_node_indices()
            .map(|idx| idx.index())
            .collect()
    }

    /// Returns just the raw ID values from the current selection as a flat list.
    /// This is the lightest possible output when you only need ID values.
    ///
    /// Returns:
    ///     List of ID values (int, str, or whatever type the IDs are)
    ///
    /// Example:
    ///     ```python
    ///     user_ids = graph.select('User').ids()
    ///     # Returns: [1, 2, 3, 4, 5, ...]
    ///     ```
    fn ids(&self) -> PyResult<Py<PyAny>> {
        Python::attach(|py| {
            let result = PyList::empty(py);

            for node_idx in self.cursor.selection.current_node_indices() {
                if let Some(node) = self.inner.get_node(node_idx) {
                    result.append(py_out::value_to_py(py, &node.id())?)?;
                }
            }

            Ok(result.into())
        })
    }

    /// Look up a single node by its type and ID value. O(1) after first call.
    ///
    /// This is much faster than select().where() for single-node lookups
    /// because it uses a hash index instead of scanning all nodes.
    ///
    /// Args:
    ///     node_type: The type of node to look up (e.g., "User", "Product")
    ///     node_id: The ID value of the node
    ///
    /// Returns:
    ///     Dict with all node properties, or None if not found
    ///
    /// Example:
    ///     ```python
    ///     user = graph.node("User", 38870)
    ///     ```
    #[pyo3(signature = (node_type, node_id))]
    fn node(&mut self, node_type: &str, node_id: &Bound<'_, PyAny>) -> PyResult<Option<Py<PyAny>>> {
        // Convert Python value to Rust Value
        let id_value = py_in::py_value_to_value(node_id)?;

        // Get mutable access to build index if needed
        let graph = Arc::make_mut(&mut self.inner);

        // This will build the index lazily if not already built
        let node_idx = match graph.lookup_by_id(node_type, &id_value) {
            Some(idx) => idx,
            None => return Ok(None),
        };

        // Get the node data
        let node = match graph.get_node(node_idx) {
            Some(n) => n,
            None => return Ok(None),
        };

        // Convert to Python dict
        let node_info = node.to_node_info(&graph.interner);
        Python::attach(|py| {
            let dict = py_out::nodeinfo_to_pydict(py, &node_info)?;
            Ok(Some(dict))
        })
    }

    /// Return True if a node of ``node_type`` with the given id exists. O(1).
    ///
    /// Uses the same id-index as ``node()`` — no scan. Mirrors ``node()``'s
    /// parameter shape and id-coercion semantics exactly (ids are integers in
    /// every storage mode; a Python ``int`` and the stored id normalize to the
    /// same key).
    ///
    /// Args:
    ///     node_type: The type of node to check (e.g., "User", "Product")
    ///     unique_id: The ID value of the node
    ///
    /// Returns:
    ///     True if a matching node exists, False otherwise
    ///
    /// Example:
    ///     ```python
    ///     if graph.exists("User", 38870):
    ///         ...
    ///     ```
    #[pyo3(signature = (node_type, unique_id))]
    fn exists(&self, node_type: &str, unique_id: &Bound<'_, PyAny>) -> PyResult<bool> {
        let id_value = py_in::py_value_to_value(unique_id)?;
        // Read-only, O(1) typed lookup. `lookup_by_id_readonly` self-heals the
        // id-index on a miss (interior mutability in `IdIndexStore`), so no
        // `&mut` / `Arc::make_mut` is needed and no per-call allocation beyond
        // the lookup key.
        Ok(self
            .inner
            .lookup_by_id_readonly(node_type, &id_value)
            .is_some())
    }

    // ========================================================================
    // Code Entity Search Methods
    // ========================================================================

    /// Find code entities by name, with disambiguation context.
    ///
    /// Searches across code entity node types (Function, Struct, Class, Enum,
    /// Trait, Protocol, Interface, Module, Constant) for nodes matching the
    /// given name or qualified_name.
    ///
    /// Args:
    ///     name: Entity name to search for (e.g. "execute", "KnowledgeGraph")
    ///     node_type: Optional filter — only search this node type
    ///         (e.g. "Function", "Struct")
    ///
    /// Returns:
    ///     List of dicts, each containing: type, name, qualified_name,
    ///     file_path, line_number, and optionally signature and visibility
    ///
    /// Example:
    ///     ```python
    ///     results = graph.find("execute")
    ///     results = graph.find("KnowledgeGraph", node_type="Struct")
    ///     ```
    #[pyo3(signature = (name, node_type=None, match_type=None))]
    fn find(
        &self,
        name: &str,
        node_type: Option<&str>,
        match_type: Option<&str>,
    ) -> PyResult<Py<PyAny>> {
        let match_type = match_type.unwrap_or("exact");
        let name_lower = name.to_lowercase();
        let types_to_search: Vec<&str> = match node_type {
            Some(nt) => vec![nt],
            None => kglite_core::api::CODE_TYPES.to_vec(),
        };

        let mut results: Vec<kglite_core::api::NodeInfo> = Vec::new();
        for nt in &types_to_search {
            if let Some(indices) = self.inner.type_indices.get(nt) {
                for idx in indices.iter() {
                    if let Some(node) = self.inner.get_node(idx) {
                        let matches = match match_type {
                            "contains" => {
                                node.field_contains_ci("name", &name_lower)
                                    || node.field_contains_ci("title", &name_lower)
                            }
                            "starts_with" => {
                                node.field_starts_with_ci("name", &name_lower)
                                    || node.field_starts_with_ci("title", &name_lower)
                            }
                            _ => {
                                // "exact" (default)
                                let name_val = Value::String(name.to_string());
                                node.get_field_ref("name")
                                    .map(|v| *v == name_val)
                                    .unwrap_or(false)
                                    || node
                                        .get_field_ref("title")
                                        .map(|v| *v == name_val)
                                        .unwrap_or(false)
                            }
                        };
                        if matches {
                            results.push(node.to_node_info(&self.inner.interner));
                        }
                    }
                }
            }
        }

        Python::attach(|py| {
            let list = PyList::empty(py);
            for node_info in &results {
                let dict = py_out::nodeinfo_to_pydict(py, node_info)?;
                list.append(dict)?;
            }
            Ok(list.into_any().unbind())
        })
    }

    /// Get the source location of one or more code entities.
    ///
    /// Resolves names or qualified names to code entities and returns
    /// file paths and line ranges. Accepts a single string or a list.
    ///
    /// Args:
    ///     name: Entity name, qualified name, or list of names.
    ///     node_type: Optional node type hint ("Function", "Struct", etc.)
    ///
    /// Returns:
    ///     Single name: dict with file_path, line_number, end_line, line_count,
    ///         name, qualified_name, type, signature.
    ///     List of names: list of dicts (one per name).
    ///     Ambiguous names return {"name": ..., "ambiguous": true, "matches": [...]}.
    ///     Unknown names return {"name": ..., "error": "Node not found: ..."}.
    ///
    /// Example:
    ///     ```python
    ///     loc = graph.source("execute_single_clause")
    ///     locs = graph.source(["KnowledgeGraph", "build", "execute"])
    ///     ```
    #[pyo3(signature = (name, node_type=None))]
    fn source(&self, name: &Bound<'_, PyAny>, node_type: Option<&str>) -> PyResult<Py<PyAny>> {
        // Check if name is a list/sequence of strings
        if let Ok(list) = name.cast::<PyList>() {
            let names: Vec<String> = list.extract()?;
            return Python::attach(|py| {
                let result = PyList::empty(py);
                for n in &names {
                    let dict = self.source_one(py, n, node_type)?;
                    result.append(dict)?;
                }
                Ok(result.into_any().unbind())
            });
        }

        // Single string
        let name_str: String = name.extract()?;
        Python::attach(|py| self.source_one(py, &name_str, node_type))
    }

    /// Get the full neighborhood of a code entity.
    ///
    /// Returns the node's properties and all related entities grouped by
    /// relationship type. If the name is ambiguous (matches multiple nodes),
    /// returns the matches so you can refine with a qualified name.
    ///
    /// Args:
    ///     name: Entity name (e.g. "build") or qualified name
    ///         (e.g. "kglite.code_tree.builder.build")
    ///     node_type: Optional node type hint ("Function", "Struct", etc.)
    ///     hops: Max traversal depth for multi-hop neighbors (default 1)
    ///
    /// Returns:
    ///     Dict with "node" (properties), "defined_in" (file path), and
    ///     relationship groups (e.g. "HAS_METHOD", "CALLS", "CALLED_BY")
    ///
    /// Example:
    ///     ```python
    ///     ctx = graph.context("KnowledgeGraph")
    ///     ctx = graph.context("kglite.code_tree.builder.build", hops=2)
    ///     ```
    #[pyo3(signature = (name, node_type=None, hops=None))]
    fn context(
        &self,
        name: &str,
        node_type: Option<&str>,
        hops: Option<usize>,
    ) -> PyResult<Py<PyAny>> {
        let hops = hops.unwrap_or(1);

        let (resolved, matches) = self.resolve_code_entity(name, node_type);

        let target_idx = match resolved {
            Some(idx) => idx,
            None => {
                return Python::attach(|py| {
                    let dict = PyDict::new(py);
                    if matches.is_empty() {
                        dict.set_item("error", format!("Node not found: {}", name))?;
                    } else {
                        dict.set_item("ambiguous", true)?;
                        let match_list = PyList::empty(py);
                        for (_, info) in &matches {
                            let d = py_out::nodeinfo_to_pydict(py, info)?;
                            match_list.append(d)?;
                        }
                        dict.set_item("matches", match_list)?;
                    }
                    Ok(dict.into_any().unbind())
                });
            }
        };

        let target_node = self.inner.get_node(target_idx).ok_or_else(|| {
            crate::error_py::kg_to_pyerr(crate::error::KgError::Argument(
                "Node disappeared".to_string(),
            ))
        })?;

        // Phase 2: Build result dict
        Python::attach(|py| {
            let result = PyDict::new(py);

            // Node properties
            let node_info = target_node.to_node_info(&self.inner.interner);
            let node_dict = py_out::nodeinfo_to_pydict(py, &node_info)?;
            result.set_item("node", &node_dict)?;

            // defined_in (file_path shortcut)
            if let Some(Value::String(fp)) = target_node.get_field_ref("file_path").as_deref() {
                result.set_item("defined_in", fp)?;
            }

            // Phase 3: Collect neighbors, grouped by edge type
            // For hops > 1, do BFS expansion
            let neighbor_indices = if hops <= 1 {
                // Direct neighbors only
                let mut neighbors = HashSet::new();
                for edge in self
                    .inner
                    .graph
                    .edges_directed(target_idx, petgraph::Direction::Outgoing)
                {
                    neighbors.insert(edge.target());
                }
                for edge in self
                    .inner
                    .graph
                    .edges_directed(target_idx, petgraph::Direction::Incoming)
                {
                    neighbors.insert(edge.source());
                }
                neighbors
            } else {
                // BFS expansion for N hops
                let mut visited = HashSet::new();
                visited.insert(target_idx);
                let mut frontier = HashSet::new();
                frontier.insert(target_idx);

                for _ in 0..hops {
                    let mut next_frontier = HashSet::new();
                    for &node in &frontier {
                        for neighbor in self.inner.graph.neighbors_undirected(node) {
                            if visited.insert(neighbor) {
                                next_frontier.insert(neighbor);
                            }
                        }
                    }
                    if next_frontier.is_empty() {
                        break;
                    }
                    frontier = next_frontier;
                }
                visited.remove(&target_idx);
                visited
            };

            // Group outgoing edges by type
            let mut outgoing_groups: HashMap<String, Vec<NodeIndex>> = HashMap::new();
            let mut incoming_groups: HashMap<String, Vec<NodeIndex>> = HashMap::new();

            for edge in self
                .inner
                .graph
                .edges_directed(target_idx, petgraph::Direction::Outgoing)
            {
                let edge_type = edge
                    .weight()
                    .connection_type_str(&self.inner.interner)
                    .to_string();
                let target = edge.target();
                if hops <= 1 || neighbor_indices.contains(&target) {
                    outgoing_groups.entry(edge_type).or_default().push(target);
                }
            }

            for edge in self
                .inner
                .graph
                .edges_directed(target_idx, petgraph::Direction::Incoming)
            {
                let edge_type = edge
                    .weight()
                    .connection_type_str(&self.inner.interner)
                    .to_string();
                let source = edge.source();
                if hops <= 1 || neighbor_indices.contains(&source) {
                    incoming_groups.entry(edge_type).or_default().push(source);
                }
            }

            // For multi-hop: also collect edges between neighbor nodes
            if hops > 1 {
                for &n_idx in &neighbor_indices {
                    for edge in self
                        .inner
                        .graph
                        .edges_directed(n_idx, petgraph::Direction::Outgoing)
                    {
                        let t = edge.target();
                        if t != target_idx && neighbor_indices.contains(&t) {
                            let edge_type = edge
                                .weight()
                                .connection_type_str(&self.inner.interner)
                                .to_string();
                            outgoing_groups.entry(edge_type).or_default().push(t);
                        }
                    }
                }
            }

            // Convert outgoing groups to Python
            for (edge_type, indices) in &outgoing_groups {
                let list = PyList::empty(py);
                let mut seen = HashSet::new();
                for &idx in indices {
                    if !seen.insert(idx) {
                        continue; // deduplicate
                    }
                    if let Some(node) = self.inner.get_node(idx) {
                        let info = node.to_node_info(&self.inner.interner);
                        let d = py_out::nodeinfo_to_pydict(py, &info)?;
                        list.append(d)?;
                    }
                }
                result.set_item(edge_type.as_str(), list)?;
            }

            // Convert incoming groups to Python (prefix with "incoming_" to avoid collision)
            for (edge_type, indices) in &incoming_groups {
                let key = if outgoing_groups.contains_key(edge_type) {
                    format!("incoming_{}", edge_type)
                } else {
                    // Use a readable reverse name for common patterns
                    match edge_type.as_str() {
                        "CALLS" => "called_by".to_string(),
                        "HAS_METHOD" => "method_of".to_string(),
                        "DEFINES" => "defined_by".to_string(),
                        "USES_TYPE" => "used_by".to_string(),
                        "IMPLEMENTS" => "implemented_by".to_string(),
                        "EXTENDS" => "extended_by".to_string(),
                        _ => format!("incoming_{}", edge_type),
                    }
                };
                let list = PyList::empty(py);
                let mut seen = HashSet::new();
                for &idx in indices {
                    if !seen.insert(idx) {
                        continue;
                    }
                    if let Some(node) = self.inner.get_node(idx) {
                        let info = node.to_node_info(&self.inner.interner);
                        let d = py_out::nodeinfo_to_pydict(py, &info)?;
                        list.append(d)?;
                    }
                }
                result.set_item(key.as_str(), list)?;
            }

            Ok(result.into_any().unbind())
        })
    }

    /// Get a table of contents for a file — all code entities defined in it.
    ///
    /// Returns entities sorted by line_number with a type summary.
    ///
    /// Args:
    ///     file_path: Path of the file (the File node's path).
    ///
    /// Returns:
    ///     Dict with "file" (path), "entities" (list of dicts sorted by
    ///     line_number, each with type, name, qualified_name, line_number,
    ///     end_line, and optionally signature), and "summary" (type -> count).
    ///     Returns {"error": "..."} if file not found.
    ///
    /// Example:
    ///     ```python
    ///     toc = graph.toc("src/graph/mod.rs")
    ///     ```
    #[pyo3(signature = (file_path))]
    fn toc(&self, file_path: &str) -> PyResult<Py<PyAny>> {
        let file_id = Value::String(file_path.to_string());

        // Find the File node by its id (path)
        let file_idx = if let Some(indices) = self.inner.type_indices.get("File") {
            indices.iter().find(|idx| {
                self.inner
                    .get_node(*idx)
                    .map(|n| *n.id() == file_id)
                    .unwrap_or(false)
            })
        } else {
            None
        };

        let file_idx = match file_idx {
            Some(idx) => idx,
            None => {
                return Python::attach(|py| {
                    let dict = PyDict::new(py);
                    dict.set_item("error", format!("File not found: {}", file_path))?;
                    Ok(dict.into_any().unbind())
                });
            }
        };

        // Collect all entities connected via outgoing DEFINES edges
        // (type, name, qualified_name, line_number, end_line, signature)
        let mut entities: Vec<(String, String, String, i64, i64, Option<String>)> = Vec::new();

        for edge in self
            .inner
            .graph
            .edges_directed(file_idx, petgraph::Direction::Outgoing)
        {
            if edge.weight().connection_type != kglite_core::api::InternedKey::from_str("DEFINES") {
                continue;
            }
            if let Some(node) = self.inner.get_node(edge.target()) {
                let node_type = node.get_node_type_ref(&self.inner.interner).to_string();
                let name = match &*node.title() {
                    Value::String(s) => s.clone(),
                    _ => String::new(),
                };
                let qname = match &*node.id() {
                    Value::String(s) => s.clone(),
                    _ => String::new(),
                };
                let line = match node.get_field_ref("line_number").as_deref() {
                    Some(Value::Int64(n)) => *n,
                    _ => 0,
                };
                let end = match node.get_field_ref("end_line").as_deref() {
                    Some(Value::Int64(n)) => *n,
                    _ => 0,
                };
                let sig = match node.get_field_ref("signature").as_deref() {
                    Some(Value::String(s)) => Some(s.clone()),
                    _ => None,
                };
                entities.push((node_type, name, qname, line, end, sig));
            }
        }

        // Sort by line_number
        entities.sort_by_key(|e| e.3);

        // Build summary: type -> count
        let mut summary: HashMap<String, usize> = HashMap::new();
        for e in &entities {
            *summary.entry(e.0.clone()).or_insert(0) += 1;
        }

        Python::attach(|py| {
            let result = PyDict::new(py);
            result.set_item("file", file_path)?;

            let entity_list = PyList::empty(py);
            for (etype, name, qname, line, end, sig) in &entities {
                let d = PyDict::new(py);
                d.set_item("type", etype)?;
                d.set_item("name", name)?;
                d.set_item("qualified_name", qname)?;
                d.set_item("line_number", line)?;
                d.set_item("end_line", end)?;
                if let Some(s) = sig {
                    d.set_item("signature", s)?;
                }
                entity_list.append(d)?;
            }
            result.set_item("entities", entity_list)?;

            let summary_dict = PyDict::new(py);
            let mut sorted_summary: Vec<_> = summary.iter().collect();
            sorted_summary.sort_by_key(|(k, _)| (*k).clone());
            for (k, v) in sorted_summary {
                summary_dict.set_item(k.as_str(), v)?;
            }
            result.set_item("summary", summary_dict)?;

            Ok(result.into_any().unbind())
        })
    }
}
