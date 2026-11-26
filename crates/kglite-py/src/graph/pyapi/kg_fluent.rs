//! KnowledgeGraph #[pymethods]: the selection and filter half of the fluent
//! chain, plus its materialisers and the code-entity lookups (the traverse /
//! compare / derive-property half is in `kg_introspection.rs`).
//!
//! PyO3 merges multiple `#[pymethods] impl` blocks at class-registration
//! time, so splitting them across files is purely structural — no runtime
//! impact.

use crate::datatypes::values::{FilterCondition, Value};
use crate::datatypes::{py_in, py_out};
use crate::graph::KnowledgeGraph;
use kglite_core::api::mutation::OperationReport;
use kglite_core::api::GraphRead;
use kglite_core::api::PlanStep;
use kglite_core::api::TemporalContext;
use petgraph::graph::NodeIndex;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3::Bound;
use std::collections::HashMap;

/// Map a core `filtering::*` error string into the fluent-API `PyErr`. Shared
/// by every selection-mutator routed through `derive_with`.
fn fluent_arg_err(e: String) -> PyErr {
    crate::error_py::kg_to_pyerr(crate::error::KgError::Argument(e))
}

#[pymethods]
impl KnowledgeGraph {
    /// Declare which two properties bound a node type's or relationship type's validity interval.
    #[pyo3(signature = (type_name, valid_from, valid_to, convention=None, source_type=None, empty_when=None))]
    // The declaration's bounds, convention, source type and empty-row option are separate keyword arguments.
    #[allow(clippy::too_many_arguments)]
    fn set_temporal(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        type_name: String,
        valid_from: String,
        valid_to: String,
        convention: Option<&str>,
        source_type: Option<String>,
        empty_when: Option<&str>,
    ) -> PyResult<()> {
        let mut guard = crate::graph::pyapi::borrow::borrow_mut_waiting(slf)?;
        let this: &mut KnowledgeGraph = &mut guard;

        use kglite_core::api::temporal::{self, TemporalTarget};
        let (convention, empty_when) =
            crate::graph::parse_interval_options(convention, empty_when)?;
        let argument = |message: String| {
            crate::error_py::kg_to_pyerr(crate::error::KgError::Argument(message))
        };
        let report = this.logged_write(|graph| {
            let is_node = graph.type_indices.contains_key(&type_name);
            let is_relationship = graph.connection_type_metadata.contains_key(&type_name);
            // A name that is both a node type and a relationship type is the node
            // type, unless a source type says otherwise.
            let target = match (is_node, is_relationship, source_type) {
                (_, true, Some(source)) => TemporalTarget::Relationship {
                    rel_type: type_name,
                    source_type: Some(source),
                },
                (_, false, Some(_)) => {
                    return Err(argument(format!(
                        "source_type applies to a relationship type, and '{type_name}' is not one"
                    )))
                }
                (true, _, None) => TemporalTarget::Node(type_name),
                (false, true, None) => TemporalTarget::Relationship {
                    rel_type: type_name,
                    source_type: None,
                },
                (false, false, None) => {
                    return Err(argument(format!(
                        "'{type_name}' is not a known node type or connection type"
                    )))
                }
            };
            temporal::declare_defaulted(
                graph,
                &target,
                &valid_from,
                &valid_to,
                (convention, empty_when),
            )
            .map_err(argument)
        })?;
        crate::graph::warn_declaration(py, &report)
    }

    /// Set the instant unprefixed statements and fluent cursors read on a graph with validity declarations.
    #[pyo3(signature = (value, persist=false))]
    fn set_valid_time_default(
        slf: &Bound<'_, Self>,
        value: &Bound<'_, PyAny>,
        persist: bool,
    ) -> PyResult<()> {
        let mut guard = crate::graph::pyapi::borrow::borrow_mut_waiting(slf)?;
        let this: &mut KnowledgeGraph = &mut guard;

        use crate::datatypes::py_in::query_date;
        use kglite_core::api::temporal::ValidTimeDefault;
        let text = value.extract::<String>().ok();
        let default = match text.as_deref().map(str::trim) {
            Some(word)
                if word.eq_ignore_ascii_case("today") || word.eq_ignore_ascii_case("all") =>
            {
                ValidTimeDefault::parse(word).map_err(fluent_arg_err)?
            }
            _ => {
                let (day, _) = query_date(value, "value").map_err(|err| {
                    if text.is_some() {
                        fluent_arg_err(format!(
                            "valid-time default {:?} is not 'today', 'all' or a date ({err})",
                            text.as_deref().unwrap_or_default()
                        ))
                    } else {
                        err
                    }
                })?;
                ValidTimeDefault::Date(day)
            }
        };
        this.unlogged_write_lazy(|inner| {
            kglite_core::api::make_dir_graph_mut_preserving_lineage(inner)
                .set_valid_time_default(default, persist)
        });
        Ok(())
    }

    /// The graph's valid-time default: 'today', 'all' or a YYYY-MM-DD date.
    fn get_valid_time_default(slf: &Bound<'_, Self>) -> PyResult<String> {
        let guard = crate::graph::pyapi::borrow::borrow_waiting(slf)?;
        let this: &KnowledgeGraph = &guard;
        Ok(this.graph().valid_time_default.to_string())
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
    fn date(
        slf: &Bound<'_, Self>,
        date_str: Option<&Bound<'_, PyAny>>,
        end_str: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let guard = crate::graph::pyapi::borrow::borrow_waiting(slf)?;
        let this: &KnowledgeGraph = &guard;

        use crate::datatypes::py_in::query_date;
        let is_all = date_str
            .and_then(|value| value.extract::<String>().ok())
            .is_some_and(|text| text == "all");
        let mut new_kg = this.clone();
        new_kg.cursor.temporal_context = match (date_str, end_str) {
            _ if is_all => TemporalContext::All,
            (Some(start), Some(end)) => {
                let (start_date, _) = query_date(start, "date_str")?;
                let (end_date, end_precision) = query_date(end, "end_str")?;
                let expanded_end =
                    kglite_core::api::timeseries::expand_end(end_date, end_precision);
                TemporalContext::During(start_date, expanded_end)
            }
            (Some(start), None) => TemporalContext::At(query_date(start, "date_str")?.0),
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

    /// Select all nodes of a given type.
    #[pyo3(signature = (node_type, sort=None, limit=None, temporal=None, include_secondary=false))]
    fn select(
        slf: &Bound<'_, Self>,
        node_type: String,
        sort: Option<&Bound<'_, PyAny>>,
        limit: Option<usize>,
        temporal: Option<bool>,
        include_secondary: bool,
    ) -> PyResult<Self> {
        let mut guard = crate::graph::pyapi::borrow::borrow_mut_waiting(slf)?;
        let this: &mut KnowledgeGraph = &mut guard;

        let mut new_kg = this.clone();

        let estimated = if include_secondary {
            this.graph().nodes_with_label(&node_type).len()
        } else {
            this.graph()
                .type_indices
                .get(&node_type)
                .map(|v| v.len())
                .unwrap_or(0)
        };
        new_kg.cursor.selection.clear_execution_plan();

        let sort_fields = if let Some(spec) = sort {
            match spec.extract::<String>() {
                Ok(field) => Some(vec![(field, true)]),
                Err(_) => Some(py_in::parse_sort_fields(spec, None)?),
            }
        } else {
            None
        };

        // A validity error — `temporal=True` on an undeclared type, a bound
        // the filter cannot read — is a `ValueError`; a seeding error an
        // `ArgumentError`.
        let valid_time = kglite_core::api::fluent::FluentFilter::for_select(
            this.graph(),
            &this.cursor.temporal_context,
            &node_type,
            temporal,
        )
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
        kglite_core::api::fluent::select_nodes(
            this.graph(),
            &mut new_kg.cursor.selection,
            &node_type,
            include_secondary,
            sort_fields,
            limit,
            &valid_time,
        )
        .map_err(|e| {
            if valid_time.finish().is_err() {
                pyo3::exceptions::PyValueError::new_err(e)
            } else {
                fluent_arg_err(e)
            }
        })?;

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

    /// Filter the current selection by property conditions.
    #[pyo3(signature = (conditions, sort=None, limit=None))]
    #[pyo3(name = "where")]
    fn where_method(
        slf: &Bound<'_, Self>,
        conditions: &Bound<'_, PyDict>,
        sort: Option<&Bound<'_, PyAny>>,
        limit: Option<usize>,
    ) -> PyResult<Self> {
        let guard = crate::graph::pyapi::borrow::borrow_waiting(slf)?;
        let this: &KnowledgeGraph = &guard;

        let filter_conditions = py_in::pydict_to_filter_conditions(conditions)?;
        let sort_fields = match sort {
            Some(spec) => Some(py_in::parse_sort_fields(spec, None)?),
            None => None,
        };

        this.derive_with(|inner, cursor| {
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
        slf: &Bound<'_, Self>,
        conditions: &Bound<'_, PyList>,
        sort: Option<&Bound<'_, PyAny>>,
        limit: Option<usize>,
    ) -> PyResult<Self> {
        let guard = crate::graph::pyapi::borrow::borrow_waiting(slf)?;
        let this: &KnowledgeGraph = &guard;

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

        this.derive_with(|inner, cursor| {
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

    /// Filter nodes based on whether they have connections.
    #[pyo3(signature = (include_orphans=None, sort=None, limit=None))]
    fn where_orphans(
        slf: &Bound<'_, Self>,
        include_orphans: Option<bool>,
        sort: Option<&Bound<'_, PyAny>>,
        limit: Option<usize>,
    ) -> PyResult<Self> {
        let guard = crate::graph::pyapi::borrow::borrow_waiting(slf)?;
        let this: &KnowledgeGraph = &guard;

        let include = include_orphans.unwrap_or(true);
        let sort_fields = if let Some(spec) = sort {
            Some(py_in::parse_sort_fields(spec, None)?)
        } else {
            None
        };

        let valid_time = kglite_core::api::fluent::FluentFilter::for_walk(
            this.graph(),
            &this.cursor.temporal_context,
            None,
            "where_orphans()",
        )
        .map_err(fluent_arg_err)?;
        this.derive_with(|inner, cursor| {
            kglite_core::api::fluent::filter_orphan_nodes(
                inner,
                &mut cursor.selection,
                include,
                sort_fields.as_ref(),
                limit,
                Some(&valid_time),
            )
            .map_err(fluent_arg_err)
        })
    }

    /// Sort the current selection.
    #[pyo3(signature = (sort, ascending=None))]
    fn sort(
        slf: &Bound<'_, Self>,
        sort: &Bound<'_, PyAny>,
        ascending: Option<bool>,
    ) -> PyResult<Self> {
        let guard = crate::graph::pyapi::borrow::borrow_waiting(slf)?;
        let this: &KnowledgeGraph = &guard;

        let sort_fields = py_in::parse_sort_fields(sort, ascending)?;
        this.derive_with(|inner, cursor| {
            kglite_core::api::fluent::sort_nodes(inner, &mut cursor.selection, sort_fields)
                .map_err(fluent_arg_err)
        })
    }

    /// Limit the number of nodes per parent group.
    fn limit(slf: &Bound<'_, Self>, max_per_group: usize) -> PyResult<Self> {
        let guard = crate::graph::pyapi::borrow::borrow_waiting(slf)?;
        let this: &KnowledgeGraph = &guard;

        this.derive_with(|inner, cursor| {
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
    fn offset(slf: &Bound<'_, Self>, n: usize) -> PyResult<Self> {
        let guard = crate::graph::pyapi::borrow::borrow_waiting(slf)?;
        let this: &KnowledgeGraph = &guard;

        this.derive_with(|inner, cursor| {
            kglite_core::api::fluent::offset_nodes(inner, &mut cursor.selection, n)
                .map_err(fluent_arg_err)
        })
    }

    /// Filter current selection to nodes that have at least one connection
    /// of the given type. Equivalent to Cypher's WHERE EXISTS {(n)-[:TYPE]->()}.
    #[pyo3(signature = (connection_type, direction=None))]
    fn where_connected(
        slf: &Bound<'_, Self>,
        connection_type: &str,
        direction: Option<&str>,
    ) -> PyResult<Self> {
        let guard = crate::graph::pyapi::borrow::borrow_waiting(slf)?;
        let this: &KnowledgeGraph = &guard;

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

        let valid_time = kglite_core::api::fluent::FluentFilter::for_walk(
            this.graph(),
            &this.cursor.temporal_context,
            Some(connection_type),
            "where_connected()",
        )
        .map_err(fluent_arg_err)?;
        this.derive_with(|inner, cursor| {
            kglite_core::api::fluent::filter_by_connection(
                inner,
                &mut cursor.selection,
                connection_type,
                dir,
                Some(&valid_time),
            )
            .map_err(fluent_arg_err)
        })
    }

    /// Keep the nodes valid at a date: each under its type's declared or named bounds.
    #[pyo3(signature = (date=None, date_from_field=None, date_to_field=None))]
    fn valid_at(
        slf: &Bound<'_, Self>,
        date: Option<&Bound<'_, PyAny>>,
        date_from_field: Option<&str>,
        date_to_field: Option<&str>,
    ) -> PyResult<Self> {
        let mut guard = crate::graph::pyapi::borrow::borrow_mut_waiting(slf)?;
        let this: &mut KnowledgeGraph = &mut guard;

        let date = match date {
            Some(d) => Some(crate::datatypes::py_in::query_date(d, "date")?.0),
            None => None,
        };
        let filter = kglite_core::api::fluent::FluentFilter::for_valid_at(
            this.graph(),
            &this.cursor.selection,
            date,
            &this.cursor.temporal_context,
            date_from_field,
            date_to_field,
        )
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
        this.retain_valid(filter, "VALID_AT")
    }

    /// Keep the nodes whose validity overlaps a range; a partial end date covers its whole period.
    #[pyo3(signature = (start_date, end_date, date_from_field=None, date_to_field=None))]
    fn valid_during(
        slf: &Bound<'_, Self>,
        start_date: &Bound<'_, PyAny>,
        end_date: &Bound<'_, PyAny>,
        date_from_field: Option<&str>,
        date_to_field: Option<&str>,
    ) -> PyResult<Self> {
        let mut guard = crate::graph::pyapi::borrow::borrow_mut_waiting(slf)?;
        let this: &mut KnowledgeGraph = &mut guard;

        let (start, _) = crate::datatypes::py_in::query_date(start_date, "start_date")?;
        let (end, precision) = crate::datatypes::py_in::query_date(end_date, "end_date")?;
        let end = kglite_core::api::timeseries::expand_end(end, precision);
        let filter = kglite_core::api::fluent::FluentFilter::for_valid_during(
            this.graph(),
            &this.cursor.selection,
            start,
            end,
            date_from_field,
            date_to_field,
        )
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
        this.retain_valid(filter, "VALID_DURING")
    }

    /// Set properties on the selected nodes of a detached copy, judged against valid-time declarations.
    #[pyo3(signature = (properties, keep_selection=None))]
    fn update(
        slf: &Bound<'_, Self>,
        properties: &Bound<'_, PyDict>,
        keep_selection: Option<bool>,
    ) -> PyResult<Py<PyAny>> {
        let mut guard = crate::graph::pyapi::borrow::borrow_mut_waiting(slf)?;
        let this: &mut KnowledgeGraph = &mut guard;

        this.check_durable_owner()?;
        let current_index = this.cursor.selection.get_level_count().saturating_sub(1);
        let level = this
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

        // Every property is judged together against the valid-time
        // declarations before anything is written, so moving an interval by
        // writing both bounds is one legal write; a refusal raises. The write
        // lands on `self.graph()`; the handle returned below is deliberately
        // detached, so `self` is what owns the durability state.
        let written = this.logged_write(|graph| {
            kglite_core::api::mutation::update_node_property_set(graph, &nodes, &parsed_properties)
                .map_err(|e| super::kg_mutation::bulk_write_err(graph, e))
        })?;
        let total_updated = written.nodes_updated;
        let errors = written.errors;
        let warnings = written.warnings;
        let diagnostics = written.diagnostics;

        let mut new_kg = this.detached_view(keep_selection.unwrap_or(false));

        let report = kglite_core::api::mutation::NodeOperationReport {
            operation_type: "update".to_string(),
            timestamp: chrono::Utc::now(),
            nodes_created: 0,
            nodes_updated: total_updated,
            nodes_skipped: 0,
            processing_time_ms: 0.0,
            errors,
            warnings,
            diagnostics,
        };
        let advisories = report.warnings.clone();

        let report_index = new_kg.add_report(OperationReport::NodeOperation(report));

        Python::attach(|py| {
            crate::graph::warn_all(py, &advisories)?;
            let dict = PyDict::new(py);
            dict.set_item("graph", Py::new(py, new_kg)?.into_any())?;
            dict.set_item("nodes_updated", total_updated)?;
            dict.set_item("report_index", report_index)?;
            Ok(dict.into())
        })
    }

    /// Materialise selected nodes as a flat ``ResultView``.
    #[pyo3(signature = (limit=None))]
    fn collect(slf: &Bound<'_, Self>, limit: Option<usize>) -> PyResult<Py<PyAny>> {
        let guard = crate::graph::pyapi::borrow::borrow_waiting(slf)?;
        let this: &KnowledgeGraph = &guard;

        let max = limit.unwrap_or(usize::MAX);
        let node_indices: Vec<petgraph::graph::NodeIndex> = this
            .cursor
            .selection
            .current_node_indices()
            .take(max)
            .collect();
        let view = crate::graph::pyapi::result_view::ResultView::from_nodes_with_graph(
            this.graph(),
            &node_indices,
        );
        Python::attach(|py| Py::new(py, view).map(|v| v.into_any()))
    }

    /// Materialise selected nodes grouped by a parent type in the traversal
    /// hierarchy. Always returns a ``dict``.
    #[pyo3(signature = (group_by, *, parent_info=false, flatten_single_parent=true, limit=None))]
    fn collect_grouped(
        slf: &Bound<'_, Self>,
        group_by: &str,
        parent_info: Option<bool>,
        flatten_single_parent: Option<bool>,
        limit: Option<usize>,
    ) -> PyResult<Py<PyAny>> {
        let guard = crate::graph::pyapi::borrow::borrow_waiting(slf)?;
        let this: &KnowledgeGraph = &guard;

        let nodes = kglite_core::api::fluent::get_nodes(
            this.graph(),
            &this.cursor.selection,
            None,
            None,
            limit,
        );
        Python::attach(|py| {
            py_out::level_nodes_to_pydict(
                py,
                &this.graph().graph,
                &nodes,
                Some(group_by),
                parent_info,
                flatten_single_parent,
            )
        })
    }

    /// Export the current selection as a pandas DataFrame.
    #[pyo3(signature = (*, include_type=true, include_id=true))]
    fn to_df(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        include_type: bool,
        include_id: bool,
    ) -> PyResult<Py<PyAny>> {
        let guard = crate::graph::pyapi::borrow::borrow_waiting(slf)?;
        let this: &KnowledgeGraph = &guard;

        let mut nodes_data: Vec<(&str, kglite_core::api::NodeView<'_>)> = Vec::new();
        for node_idx in this.cursor.selection.current_node_indices() {
            if let Some(node) = this.graph().node_view(node_idx) {
                nodes_data.push((node.node_type_str(&this.graph().interner), node));
            }
        }

        // Columns emitted below straight from the node's canonical identity.
        // A stored property of the same name must be dropped from the property
        // set: the dict backing the DataFrame is keyed by column name, so a
        // duplicate does not preserve both values — it overwrites the
        // canonical one and yields a non-unique header that `to_parquet`
        // rejects outright. A canonical column the caller opted *out* of is
        // absent from this list, so a property under that name still survives.
        let mut emitted_identity: Vec<&str> = vec!["title"];
        if include_type {
            emitted_identity.push("type");
        }
        if include_id {
            emitted_identity.push("id");
        }

        let discover = |nodes: &Vec<(&str, kglite_core::api::NodeView<'_>)>| {
            kglite_core::api::discover_property_keys_excluding(
                nodes,
                &this.graph().interner,
                &emitted_identity,
            )
        };
        // Fast path: past ~50 single-typed nodes, take the keys from the
        // TypeSchema instead of scanning every node.
        let prop_keys: Vec<String> = if nodes_data.len() > 50 {
            let first_type = nodes_data[0].0;
            let all_same = nodes_data.iter().all(|(nt, _)| *nt == first_type);
            if all_same {
                kglite_core::api::schema_property_keys(this.graph(), first_type, &emitted_identity)
                    .unwrap_or_else(|| discover(&nodes_data))
            } else {
                discover(&nodes_data)
            }
        } else {
            discover(&nodes_data)
        };

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

        let prop_cols: Vec<pyo3::Bound<'_, PyList>> =
            prop_keys.iter().map(|_| PyList::empty(py)).collect();

        for (node_type, node) in &nodes_data {
            title_col.append(py_out::graph_value_to_py(
                py,
                &this.graph().graph,
                &node.title(),
            )?)?;
            if let Some(ref tc) = type_col {
                tc.append(*node_type)?;
            }
            if let Some(ref ic) = id_col {
                ic.append(py_out::value_to_py(py, &node.id())?)?;
            }
            for (j, key) in prop_keys.iter().enumerate() {
                let val = node.get_property(key);
                let val_ref = val.as_deref().unwrap_or(&Value::Null);
                prop_cols[j].append(py_out::graph_value_to_py(
                    py,
                    &this.graph().graph,
                    val_ref,
                )?)?;
            }
        }

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

        crate::datatypes::pandas_out::dataframe(py, dict.as_any(), Some(&columns), None, None)
    }

    /// Format the current selection as a human-readable string.
    ///
    /// Each node is printed as a block with type, id, title, and all properties.
    /// The ``limit`` parameter caps the number of nodes shown (default 50).
    #[pyo3(signature = (limit=50))]
    fn to_str(slf: &Bound<'_, Self>, limit: usize) -> PyResult<String> {
        let guard = crate::graph::pyapi::borrow::borrow_waiting(slf)?;
        let this: &KnowledgeGraph = &guard;

        use crate::datatypes::values::format_value;

        let node_indices: Vec<_> = this.cursor.selection.current_node_indices().collect();
        let total = node_indices.len();
        let show = total.min(limit);

        if total == 0 {
            return Ok("(empty selection)".to_string());
        }

        let mut buf = String::with_capacity(show * 200);

        for (i, &idx) in node_indices.iter().take(show).enumerate() {
            if let Some(node) = this.graph().node_view(idx) {
                if i > 0 {
                    buf.push('\n');
                }
                buf.push_str(&format!(
                    "[{}] {} (id: {})\n",
                    node.node_type_str(&this.graph().interner),
                    format_value(&node.title()),
                    format_value(&node.id()),
                ));
                // Sort property keys for deterministic output
                let mut keys: Vec<&str> = node.property_keys(&this.graph().interner);
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
    ///     print(graph.select("Initiative").show(["id", "title"]))
    ///     # Initiative(123, Juniper)
    ///     # Initiative(456, Tundra)
    ///
    ///     print(graph.select("Initiative")
    ///         .traverse("HAS_DEPOSIT_PROPOSAL")
    ///         .traverse("TESTED_BY_SITE")
    ///         .show(["id", "title"]))
    ///     # Initiative(123, Juniper) -> Proposal(456, Alpha) -> Site(789, W1)
    ///     ```
    #[pyo3(signature = (columns=None, limit=200))]
    fn show(slf: &Bound<'_, Self>, columns: Option<Vec<String>>, limit: usize) -> PyResult<String> {
        let guard = crate::graph::pyapi::borrow::borrow_waiting(slf)?;
        let this: &KnowledgeGraph = &guard;

        use kglite_core::api::fluent::format_value_compact;

        let columns = columns.unwrap_or_else(|| vec!["id".to_string(), "title".to_string()]);
        let level_count = this.cursor.selection.get_level_count();

        let fmt_node = |idx: NodeIndex| -> String {
            let node = match this.graph().node_view(idx) {
                Some(n) => n,
                None => return "?".to_string(),
            };
            let mut s = String::with_capacity(64);
            let node_type_str = node.node_type_str(&this.graph().interner);
            s.push_str(node_type_str);
            s.push('(');
            let mut first = true;
            for col in &columns {
                let resolved = this.graph().resolve_alias(node_type_str, col);
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
            let nodes: Vec<_> = this.cursor.selection.current_node_indices().collect();
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
            let level0 = this.cursor.selection.get_level(0).ok_or_else(|| {
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
                        chains.push(chain);
                        continue;
                    }

                    let level = match this.cursor.selection.get_level(level_idx) {
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
    fn py_len(slf: &Bound<'_, Self>) -> PyResult<usize> {
        let guard = crate::graph::pyapi::borrow::borrow_waiting(slf)?;
        let this: &KnowledgeGraph = &guard;
        Ok(if this.cursor.selection.has_active_selection() {
            this.cursor.selection.current_node_count()
        } else {
            this.graph().graph.node_count()
        })
    }

    fn __len__(slf: &Bound<'_, Self>) -> PyResult<usize> {
        Self::py_len(slf)
    }

    /// Returns the raw node indices in the current selection.
    /// Much faster than collect() when you only need indices for further processing.
    ///
    /// Example:
    ///     ```python
    ///     indices = graph.select('User').indices()
    ///     ```
    fn indices(slf: &Bound<'_, Self>) -> PyResult<Vec<usize>> {
        let guard = crate::graph::pyapi::borrow::borrow_waiting(slf)?;
        let this: &KnowledgeGraph = &guard;
        Ok(this
            .cursor
            .selection
            .current_node_indices()
            .map(|idx| idx.index())
            .collect())
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
    fn ids(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let guard = crate::graph::pyapi::borrow::borrow_waiting(slf)?;
        let this: &KnowledgeGraph = &guard;

        Python::attach(|py| {
            let result = PyList::empty(py);

            for node_idx in this.cursor.selection.current_node_indices() {
                if let Some(node) = this.graph().node_view(node_idx) {
                    result.append(py_out::value_to_py(py, &node.id())?)?;
                }
            }

            Ok(result.into())
        })
    }

    /// Look up a node by type and integer or string ID through the identity index.
    #[pyo3(signature = (node_type, node_id))]
    fn node(
        slf: &Bound<'_, Self>,
        node_type: &str,
        node_id: &Bound<'_, PyAny>,
    ) -> PyResult<Option<Py<PyAny>>> {
        let guard = crate::graph::pyapi::borrow::borrow_waiting(slf)?;
        let this: &KnowledgeGraph = &guard;

        let id_value = py_in::py_value_to_value(node_id)?;

        // Read-only typed index lookup — same path as `exists()`.
        // `lookup_by_id_readonly` self-heals the id-index on a miss
        // (interior mutability in `IdIndexStore`), so no `&mut` /
        // `Arc::make_mut` is needed. The old `Arc::make_mut` route
        // deep-copied the ENTIRE graph whenever any other handle
        // (a fluent clone, a frozen view, a session) shared the Arc —
        // an O(graph) hit on a point lookup.
        let node_idx = match this.graph().lookup_by_id_readonly(node_type, &id_value) {
            Some(idx) => idx,
            None => return Ok(None),
        };

        let node = match this.graph().node_view(node_idx) {
            Some(n) => n,
            None => return Ok(None),
        };

        let node_info = node.to_node_info(&this.graph().interner);
        Python::attach(|py| {
            let dict = py_out::nodeinfo_to_pydict(py, &this.graph().graph, &node_info)?;
            Ok(Some(dict))
        })
    }

    /// Return whether the identity index contains this type and integer or string ID.
    #[pyo3(signature = (node_type, unique_id))]
    fn exists(
        slf: &Bound<'_, Self>,
        node_type: &str,
        unique_id: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        let guard = crate::graph::pyapi::borrow::borrow_waiting(slf)?;
        let this: &KnowledgeGraph = &guard;

        let id_value = py_in::py_value_to_value(unique_id)?;
        // Read-only typed index lookup — see `node()` for why this must not
        // route through `Arc::make_mut`.
        Ok(this
            .graph()
            .lookup_by_id_readonly(node_type, &id_value)
            .is_some())
    }

    /// Find code entities by name, with disambiguation context.
    ///
    /// Searches across code entity node types (Function, Struct, Class, Mixin,
    /// Enum, Trait, Protocol, Interface, Module, Constant) for nodes matching
    /// the given name or qualified_name.
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
        slf: &Bound<'_, Self>,
        name: &str,
        node_type: Option<&str>,
        match_type: Option<&str>,
    ) -> PyResult<Py<PyAny>> {
        let guard = crate::graph::pyapi::borrow::borrow_waiting(slf)?;
        let this: &KnowledgeGraph = &guard;

        let match_type = match match_type.unwrap_or("exact") {
            "contains" => kglite_core::api::code_entities::CodeEntityMatch::Contains,
            "starts_with" => kglite_core::api::code_entities::CodeEntityMatch::StartsWith,
            _ => kglite_core::api::code_entities::CodeEntityMatch::Exact,
        };
        let results = kglite_core::api::code_entities::find_code_entities(
            this.graph(),
            name,
            node_type,
            match_type,
        );

        Python::attach(|py| {
            let list = PyList::empty(py);
            for node_info in &results {
                let dict = py_out::nodeinfo_to_pydict(py, &this.graph().graph, node_info)?;
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
    fn source(
        slf: &Bound<'_, Self>,
        name: &Bound<'_, PyAny>,
        node_type: Option<&str>,
    ) -> PyResult<Py<PyAny>> {
        let guard = crate::graph::pyapi::borrow::borrow_waiting(slf)?;
        let this: &KnowledgeGraph = &guard;

        if let Ok(list) = name.cast::<PyList>() {
            let names: Vec<String> = list.extract()?;
            return Python::attach(|py| {
                let result = PyList::empty(py);
                for n in &names {
                    let dict = this.source_one(py, n, node_type)?;
                    result.append(dict)?;
                }
                Ok(result.into_any().unbind())
            });
        }

        let name_str: String = name.extract()?;
        Python::attach(|py| this.source_one(py, &name_str, node_type))
    }

    /// Get the full neighborhood of a code entity.
    ///
    /// Returns the node's properties and all related entities grouped by
    /// relationship type. If the name is ambiguous (matches multiple nodes),
    /// returns the matches so you can refine with a qualified name.
    ///
    /// Args:
    ///     name: Entity name (e.g. "build") or qualified name
    ///         (e.g. "mypkg.parser.parse_file")
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
    ///     ctx = graph.context("mypkg.parser.parse_file", hops=2)
    ///     ```
    #[pyo3(signature = (name, node_type=None, hops=None))]
    fn context(
        slf: &Bound<'_, Self>,
        name: &str,
        node_type: Option<&str>,
        hops: Option<usize>,
    ) -> PyResult<Py<PyAny>> {
        let guard = crate::graph::pyapi::borrow::borrow_waiting(slf)?;
        let this: &KnowledgeGraph = &guard;

        let lookup = kglite_core::api::code_entities::code_entity_context(
            this.graph(),
            name,
            node_type,
            hops.unwrap_or(1),
        );
        Python::attach(|py| {
            let result = PyDict::new(py);
            let context = match lookup {
                kglite_core::api::code_entities::CodeContextLookup::NotFound => {
                    result.set_item("error", format!("Node not found: {}", name))?;
                    return Ok(result.into_any().unbind());
                }
                kglite_core::api::code_entities::CodeContextLookup::Ambiguous(matches) => {
                    result.set_item("ambiguous", true)?;
                    let match_list = PyList::empty(py);
                    for info in &matches {
                        match_list.append(py_out::nodeinfo_to_pydict(
                            py,
                            &this.graph().graph,
                            info,
                        )?)?;
                    }
                    result.set_item("matches", match_list)?;
                    return Ok(result.into_any().unbind());
                }
                kglite_core::api::code_entities::CodeContextLookup::Found(context) => context,
            };
            result.set_item(
                "node",
                py_out::nodeinfo_to_pydict(py, &this.graph().graph, &context.node)?,
            )?;
            if let Some(path) = &context.defined_in {
                result.set_item("defined_in", path)?;
            }
            for (edge_type, nodes) in &context.outgoing {
                let list = PyList::empty(py);
                for info in nodes {
                    list.append(py_out::nodeinfo_to_pydict(py, &this.graph().graph, info)?)?;
                }
                result.set_item(edge_type.as_str(), list)?;
            }
            for (edge_type, nodes) in &context.incoming {
                let key = if context.outgoing.contains_key(edge_type) {
                    format!("incoming_{}", edge_type)
                } else {
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
                for info in nodes {
                    list.append(py_out::nodeinfo_to_pydict(py, &this.graph().graph, info)?)?;
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
    fn toc(slf: &Bound<'_, Self>, file_path: &str) -> PyResult<Py<PyAny>> {
        let guard = crate::graph::pyapi::borrow::borrow_waiting(slf)?;
        let this: &KnowledgeGraph = &guard;

        let file_id = Value::String(file_path.to_string());

        let file_idx = if let Some(indices) = this.graph().type_indices.get("File") {
            indices.iter().find(|idx| {
                this.graph()
                    .node_view(*idx)
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

        // Tuple layout: (type, name, qualified_name, line_number, end_line,
        // signature).
        let mut entities: Vec<(String, String, String, i64, i64, Option<String>)> = Vec::new();

        for edge in this
            .graph()
            .graph
            .edges_directed(file_idx, petgraph::Direction::Outgoing)
        {
            if edge.connection_type() != kglite_core::api::InternedKey::from_str("DEFINES") {
                continue;
            }
            if let Some(node) = this.graph().node_view(edge.target()) {
                let node_type = node.get_node_type_ref(&this.graph().interner).to_string();
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

        entities.sort_by_key(|e| e.3);

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
