use kglite::api::algorithms::{leiden_communities, CommunityOptions};
use kglite::api::storage::{new_dir_graph_in_mode, StorageMode};

#[test]
fn leiden_is_callable_through_the_sealed_api_facade() {
    let graph = new_dir_graph_in_mode(StorageMode::Memory, None).expect("create graph");
    let result = leiden_communities(&graph, &CommunityOptions::default()).expect("run Leiden");

    assert_eq!(result.num_communities, 0);
    assert!(result.assignments.is_empty());
}
