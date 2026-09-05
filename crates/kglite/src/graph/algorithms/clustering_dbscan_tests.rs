use super::{dbscan, euclidean_distance_matrix};
use crate::graph::algorithms::Interrupt;
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};

fn assignments(matrix: &[Vec<f64>], eps: f64, minimum: usize) -> Vec<(usize, i64)> {
    dbscan(matrix, eps, minimum, Interrupt::default())
        .into_iter()
        .map(|a| (a.index, a.cluster))
        .collect()
}

fn expected(labels: &[i64]) -> Vec<(usize, i64)> {
    labels.iter().copied().enumerate().collect()
}

fn links(n: usize, edges: &[(usize, usize)]) -> Vec<Vec<f64>> {
    let mut matrix = vec![vec![10.0; n]; n];
    for (i, row) in matrix.iter_mut().enumerate() {
        row[i] = 0.0;
    }
    for &(a, b) in edges {
        matrix[a][b] = 1.0;
        matrix[b][a] = 1.0;
    }
    matrix
}

#[test]
fn dbscan_fifo_expands_beyond_seed_neighbors() {
    // Two core cliques linked by 3--4 require queue pushes beyond the first seed.
    let mut edges = vec![(3, 4)];
    for start in [0, 4] {
        for a in start..start + 4 {
            for b in a + 1..start + 4 {
                edges.push((a, b));
            }
        }
    }
    assert_eq!(
        assignments(&links(9, &edges), 1.0, 3),
        expected(&[0, 0, 0, 0, 0, 0, 0, 0, -1])
    );
}

#[test]
fn dbscan_noise_border_keeps_first_cluster_and_does_not_expand() {
    // Point 0 starts as noise, borders both core cliques, and belongs to the
    // first clique reached in input order; it must not bridge those clusters.
    let mut edges = vec![(0, 1), (0, 5)];
    for start in [1, 5] {
        for a in start..start + 4 {
            for b in a + 1..start + 4 {
                edges.push((a, b));
            }
        }
    }
    assert_eq!(
        assignments(&links(9, &edges), 1.0, 3),
        expected(&[0, 0, 0, 0, 0, 1, 1, 1, 1])
    );
    let permutation = [0, 5, 6, 7, 8, 1, 2, 3, 4];
    let original = links(9, &edges);
    let reversed: Vec<Vec<f64>> = permutation
        .iter()
        .map(|&a| permutation.iter().map(|&b| original[a][b]).collect())
        .collect();
    assert_eq!(
        assignments(&reversed, 1.0, 3),
        expected(&[0, 0, 0, 0, 0, 1, 1, 1, 1])
    );
}

#[test]
fn dbscan_dense_duplicates_and_multiple_seeds_have_exact_order() {
    let features = vec![
        vec![0.0],
        vec![0.0],
        vec![0.0],
        vec![8.0],
        vec![8.0],
        vec![8.0],
        vec![20.0],
    ];
    let matrix = euclidean_distance_matrix(&features, Interrupt::default());
    assert_eq!(
        assignments(&matrix, 0.0, 2),
        expected(&[0, 0, 0, 1, 1, 1, -1])
    );
    assert_eq!(
        assignments(&vec![vec![0.0; 128]; 128], 0.0, 127),
        expected(&[0; 128])
    );
    assert_eq!(
        assignments(&vec![vec![0.0; 128]; 128], 0.0, 128),
        expected(&[-1; 128])
    );
}

#[test]
fn dbscan_empty_singleton_and_other_neighbor_minimum() {
    assert!(assignments(&[], 1.0, 0).is_empty());
    assert_eq!(assignments(&[vec![0.0]], 0.0, 0), expected(&[0]));
    assert_eq!(assignments(&[vec![0.0]], f64::INFINITY, 1), expected(&[-1]));
    let pair = vec![vec![0.0, 1.0], vec![1.0, 0.0]];
    assert_eq!(assignments(&pair, 1.0, 1), expected(&[0, 0]));
    assert_eq!(assignments(&pair, 1.0, 2), expected(&[-1, -1]));
    assert_eq!(assignments(&pair, 0.0, 0), expected(&[0, 1]));
}

#[test]
fn dbscan_epsilon_boundary_and_nonfinite_values_are_unchanged() {
    let pair = vec![vec![0.0, 1.0], vec![1.0, 0.0]];
    assert_eq!(assignments(&pair, 1.0, 1), expected(&[0, 0]));
    assert_eq!(
        assignments(&pair, f64::from_bits(1.0f64.to_bits() - 1), 1),
        expected(&[-1, -1])
    );
    assert_eq!(
        assignments(&pair, f64::from_bits(1.0f64.to_bits() + 1), 1),
        expected(&[0, 0])
    );
    for eps in [-1.0, f64::NEG_INFINITY, f64::NAN] {
        assert_eq!(assignments(&pair, eps, 1), expected(&[-1, -1]));
        assert_eq!(assignments(&pair, eps, 0), expected(&[0, 1]));
    }
    assert_eq!(assignments(&pair, f64::INFINITY, 1), expected(&[0, 0]));
    let nonfinite = vec![
        vec![0.0, f64::INFINITY, f64::NAN],
        vec![f64::INFINITY, 0.0, 1.0],
        vec![f64::NAN, 1.0, 0.0],
    ];
    assert_eq!(
        assignments(&nonfinite, f64::INFINITY, 1),
        expected(&[0, 0, 0])
    );
}

#[test]
fn dbscan_interrupt_returns_no_partial_assignments() {
    static CANCELLED: AtomicBool = AtomicBool::new(true);
    let matrix = vec![vec![0.0; 128]; 128];
    let interrupts = [
        Interrupt {
            deadline: None,
            cancel: Some(&CANCELLED),
        },
        Interrupt::from_deadline(Some(Instant::now() - Duration::from_secs(1))),
    ];
    for interrupt in interrupts {
        assert!(dbscan(&matrix, 0.0, 2, interrupt).is_empty());
    }
}
