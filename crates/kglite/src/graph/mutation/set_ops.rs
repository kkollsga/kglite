// src/graph/set_ops.rs
//! Set operations on graph selections (union, intersection, difference)

use crate::graph::schema::CurrentSelection;
use petgraph::graph::NodeIndex;
use std::collections::HashSet;

/// Perform union of two selections - combines all nodes from both selections
pub fn union_selections(
    target: &mut CurrentSelection,
    source: &CurrentSelection,
) -> Result<(), String> {
    let target_level_idx = target.get_level_count().saturating_sub(1);
    let source_level_idx = source.get_level_count().saturating_sub(1);

    let target_level = target
        .get_level(target_level_idx)
        .ok_or_else(|| "No valid target selection level".to_string())?;
    let source_level = source
        .get_level(source_level_idx)
        .ok_or_else(|| "No valid source selection level".to_string())?;

    // Collect all nodes from both selections using HashSet for deduplication
    let mut all_nodes: HashSet<NodeIndex> = target_level.iter_node_indices().collect();
    all_nodes.extend(source_level.iter_node_indices());

    // Update target selection with union result
    let target_level_mut = target
        .get_level_mut(target_level_idx)
        .ok_or_else(|| "Failed to get mutable target level".to_string())?;

    target_level_mut.selections.clear();
    target_level_mut.add_selection(None, all_nodes.into_iter().collect());

    Ok(())
}

/// Perform intersection of two selections - keeps only nodes present in both
pub fn intersection_selections(
    target: &mut CurrentSelection,
    source: &CurrentSelection,
) -> Result<(), String> {
    let target_level_idx = target.get_level_count().saturating_sub(1);
    let source_level_idx = source.get_level_count().saturating_sub(1);

    let target_level = target
        .get_level(target_level_idx)
        .ok_or_else(|| "No valid target selection level".to_string())?;
    let source_level = source
        .get_level(source_level_idx)
        .ok_or_else(|| "No valid source selection level".to_string())?;

    // Build HashSets for efficient intersection
    let target_set: HashSet<NodeIndex> = target_level.iter_node_indices().collect();
    let source_set: HashSet<NodeIndex> = source_level.iter_node_indices().collect();

    // Compute intersection
    let result: Vec<NodeIndex> = target_set.intersection(&source_set).copied().collect();

    // Update target selection with intersection result
    let target_level_mut = target
        .get_level_mut(target_level_idx)
        .ok_or_else(|| "Failed to get mutable target level".to_string())?;

    target_level_mut.selections.clear();
    target_level_mut.add_selection(None, result);

    Ok(())
}

/// Perform difference of two selections - keeps nodes in target but not in source
pub fn difference_selections(
    target: &mut CurrentSelection,
    source: &CurrentSelection,
) -> Result<(), String> {
    let target_level_idx = target.get_level_count().saturating_sub(1);
    let source_level_idx = source.get_level_count().saturating_sub(1);

    let target_level = target
        .get_level(target_level_idx)
        .ok_or_else(|| "No valid target selection level".to_string())?;
    let source_level = source
        .get_level(source_level_idx)
        .ok_or_else(|| "No valid source selection level".to_string())?;

    // Build HashSets for efficient difference
    let target_set: HashSet<NodeIndex> = target_level.iter_node_indices().collect();
    let source_set: HashSet<NodeIndex> = source_level.iter_node_indices().collect();

    // Compute difference (nodes in target but not in source)
    let result: Vec<NodeIndex> = target_set.difference(&source_set).copied().collect();

    // Update target selection with difference result
    let target_level_mut = target
        .get_level_mut(target_level_idx)
        .ok_or_else(|| "Failed to get mutable target level".to_string())?;

    target_level_mut.selections.clear();
    target_level_mut.add_selection(None, result);

    Ok(())
}

/// Perform symmetric difference of two selections - keeps nodes in either but not both
pub fn symmetric_difference_selections(
    target: &mut CurrentSelection,
    source: &CurrentSelection,
) -> Result<(), String> {
    let target_level_idx = target.get_level_count().saturating_sub(1);
    let source_level_idx = source.get_level_count().saturating_sub(1);

    let target_level = target
        .get_level(target_level_idx)
        .ok_or_else(|| "No valid target selection level".to_string())?;
    let source_level = source
        .get_level(source_level_idx)
        .ok_or_else(|| "No valid source selection level".to_string())?;

    // Build HashSets for efficient symmetric difference
    let target_set: HashSet<NodeIndex> = target_level.iter_node_indices().collect();
    let source_set: HashSet<NodeIndex> = source_level.iter_node_indices().collect();

    // Compute symmetric difference (nodes in either but not both)
    let result: Vec<NodeIndex> = target_set
        .symmetric_difference(&source_set)
        .copied()
        .collect();

    // Update target selection with symmetric difference result
    let target_level_mut = target
        .get_level_mut(target_level_idx)
        .ok_or_else(|| "Failed to get mutable target level".to_string())?;

    target_level_mut.selections.clear();
    target_level_mut.add_selection(None, result);

    Ok(())
}
