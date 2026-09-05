use super::{expand_dbscan, ClusterAssignment};
use crate::graph::algorithms::Interrupt;
use crate::graph::features::spatial::geodesic_distance;

/// Preserve the matrix route's WGS84 threshold and neighbor order without its
/// dense distance allocation. Interrupted construction never reaches expansion.
pub(crate) fn geographic_dbscan(
    points: &[(f64, f64)],
    eps: f64,
    min_points: usize,
    interrupt: Interrupt,
) -> Vec<ClusterAssignment> {
    let Some(neighbors) = geographic_neighbors(points, eps, || interrupt.exceeded()) else {
        return Vec::new();
    };
    expand_dbscan(&neighbors, min_points, interrupt)
}

fn geographic_neighbors(
    points: &[(f64, f64)],
    eps: f64,
    mut interrupted: impl FnMut() -> bool,
) -> Option<Vec<Vec<usize>>> {
    if interrupted() {
        return None;
    }
    let mut neighbors = vec![Vec::new(); points.len()];
    let mut pairs_since_poll = 0usize;
    for i in 0..points.len() {
        for j in (i + 1)..points.len() {
            let distance = geodesic_distance(points[i].0, points[i].1, points[j].0, points[j].1);
            if distance <= eps {
                // Earlier indices arrive before a row's own later-index pass.
                neighbors[i].push(j);
                neighbors[j].push(i);
            }
            pairs_since_poll = (pairs_since_poll + 1) & 0x3FF;
            if pairs_since_poll == 0 && interrupted() {
                return None;
            }
        }
    }
    Some(neighbors)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::algorithms::clustering::{dbscan, haversine_distance_matrix};
    use std::sync::atomic::AtomicBool;
    use std::time::{Duration, Instant};

    fn labels(assignments: Vec<ClusterAssignment>) -> Vec<(usize, i64)> {
        assignments
            .into_iter()
            .map(|a| (a.index, a.cluster))
            .collect()
    }

    #[test]
    fn geographic_neighbors_match_wgs84_matrix_in_exact_order() {
        let mut points = vec![
            (0.0, 179.999),
            (0.0, -179.999),
            (89.999, 0.0),
            (89.999, 180.0),
            (-89.999, 0.0),
            (0.0, 0.0),
            (0.0, 179.999999),
            (12.3, 45.6),
            (12.3, 45.6),
        ];
        for _ in 0..2 {
            let matrix = haversine_distance_matrix(&points, Interrupt::default());
            let boundary = matrix[0][1];
            let below = if boundary == 0.0 {
                -f64::from_bits(1)
            } else {
                f64::from_bits(boundary.to_bits() - 1)
            };
            let thresholds = [
                0.0,
                3.0,
                boundary,
                below,
                f64::from_bits(boundary.to_bits() + 1),
                20_100_000.0,
                -1.0,
                f64::NEG_INFINITY,
                f64::INFINITY,
                f64::NAN,
            ];
            for eps in thresholds {
                let expected: Vec<Vec<usize>> = matrix
                    .iter()
                    .enumerate()
                    .map(|(i, row)| {
                        (0..row.len())
                            .filter(|&j| i != j && row[j] <= eps)
                            .collect()
                    })
                    .collect();
                assert_eq!(geographic_neighbors(&points, eps, || false), Some(expected));
                for minimum in [0, 1, 2, points.len()] {
                    assert_eq!(
                        labels(geographic_dbscan(
                            &points,
                            eps,
                            minimum,
                            Interrupt::default()
                        )),
                        labels(dbscan(&matrix, eps, minimum, Interrupt::default())),
                    );
                }
            }
            points.reverse();
        }
    }

    #[test]
    fn geographic_clusters_have_absolute_labels_and_input_indices() {
        let points = [
            (0.0, 0.0),
            (0.0, 0.00001),
            (0.0, 0.00002),
            (0.0, 1.0),
            (0.0, 1.00001),
            (0.0, 1.00002),
            (20.0, 20.0),
        ];
        let expected: Vec<_> = [0, 0, 0, 1, 1, 1, -1].into_iter().enumerate().collect();
        assert_eq!(
            labels(geographic_dbscan(&points, 3.0, 2, Interrupt::default())),
            expected
        );
        let duplicates = vec![(59.91, 10.75); 64];
        let neighbors = geographic_neighbors(&duplicates, 0.0, || false).unwrap();
        for (i, row) in neighbors.iter().enumerate() {
            assert_eq!(*row, (0..64).filter(|&j| i != j).collect::<Vec<_>>());
        }
    }

    #[test]
    fn geographic_empty_singleton_and_exact_distance_boundary() {
        assert!(geographic_dbscan(&[], 1.0, 0, Interrupt::default()).is_empty());
        assert_eq!(
            labels(geographic_dbscan(
                &[(0.0, 0.0)],
                0.0,
                0,
                Interrupt::default()
            )),
            vec![(0, 0)]
        );
        assert_eq!(
            labels(geographic_dbscan(
                &[(0.0, 0.0)],
                0.0,
                1,
                Interrupt::default()
            )),
            vec![(0, -1)]
        );
        let pair = [(0.0, 179.999), (0.0, -179.999)];
        let distance = geodesic_distance(pair[0].0, pair[0].1, pair[1].0, pair[1].1);
        assert_eq!(
            labels(geographic_dbscan(&pair, distance, 1, Interrupt::default())),
            vec![(0, 0), (1, 0)]
        );
        assert_eq!(
            labels(geographic_dbscan(
                &pair,
                f64::from_bits(distance.to_bits() - 1),
                1,
                Interrupt::default()
            )),
            vec![(0, -1), (1, -1)]
        );
    }

    #[test]
    fn geographic_mid_construction_interrupt_discards_partial_neighbors() {
        let points = vec![(59.91, 10.75); 65];
        let mut polls = 0;
        let neighbors = geographic_neighbors(&points, 0.0, || {
            polls += 1;
            polls == 2
        });
        // The second poll follows 1024 real pairs; more pairs remain unbuilt.
        assert_eq!(polls, 2);
        assert!(neighbors.is_none());
        let complete = geographic_neighbors(&points, 0.0, || false).unwrap();
        assert!(complete.iter().all(|row| row.len() == 64));
    }

    #[test]
    fn geographic_interrupt_never_returns_partial_assignments() {
        static CANCELLED: AtomicBool = AtomicBool::new(true);
        let points = [(0.0, 0.0), (0.0, 0.00001), (0.0, 0.00002)];
        let interrupts = [
            Interrupt {
                deadline: None,
                cancel: Some(&CANCELLED),
            },
            Interrupt::from_deadline(Some(Instant::now() - Duration::from_secs(1))),
        ];
        for interrupt in interrupts {
            assert!(geographic_dbscan(&points, 3.0, 2, interrupt).is_empty());
        }
        assert_eq!(geographic_neighbors(&points, 3.0, || true), None);
    }
}
