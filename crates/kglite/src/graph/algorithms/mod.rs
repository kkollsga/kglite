//! Graph algorithms — the scoring / analytics side of the codebase.
//!
//! PageRank, centrality, components, shortest path, clustering, vector
//! search. Distinguishes from `query/` which answers "which nodes match
//! this pattern?"; algorithms here answer "what does the graph look
//! like structurally?".

pub mod clustering;
pub mod graph_algorithms;
pub mod vector;
