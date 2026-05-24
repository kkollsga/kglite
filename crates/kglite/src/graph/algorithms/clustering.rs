// src/graph/clustering.rs
//
// General-purpose clustering algorithms for numeric feature vectors.
// Used by CALL cluster() in the Cypher executor.

/// A clustering assignment: original index in input array → cluster label.
pub struct ClusterAssignment {
    pub index: usize,
    pub cluster: i64, // -1 for noise (DBSCAN)
}

// ── Distance matrices ──────────────────────────────────────────────────────

/// Compute pairwise Euclidean distance matrix from feature vectors.
pub fn euclidean_distance_matrix(features: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = features.len();
    let mut dist = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let d = euclidean_distance(&features[i], &features[j]);
            dist[i][j] = d;
            dist[j][i] = d;
        }
    }
    dist
}

/// Compute pairwise Haversine (geodesic) distance matrix.
/// Each point is (lat, lon) in degrees. Returns distances in meters.
pub fn haversine_distance_matrix(points: &[(f64, f64)]) -> Vec<Vec<f64>> {
    let n = points.len();
    let mut dist = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let d = crate::graph::features::spatial::geodesic_distance(
                points[i].0,
                points[i].1,
                points[j].0,
                points[j].1,
            );
            dist[i][j] = d;
            dist[j][i] = d;
        }
    }
    dist
}

fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}

// ── Normalization ──────────────────────────────────────────────────────────

/// Normalize feature vectors to [0, 1] range per dimension (min-max scaling).
/// Modifies features in-place. Dimensions with zero range are set to 0.
pub fn normalize_features(features: &mut [Vec<f64>]) {
    if features.is_empty() {
        return;
    }
    let dims = features[0].len();
    for d in 0..dims {
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        for f in features.iter() {
            if f[d] < min {
                min = f[d];
            }
            if f[d] > max {
                max = f[d];
            }
        }
        let range = max - min;
        if range > 0.0 {
            for f in features.iter_mut() {
                f[d] = (f[d] - min) / range;
            }
        } else {
            for f in features.iter_mut() {
                f[d] = 0.0;
            }
        }
    }
}

// ── DBSCAN ─────────────────────────────────────────────────────────────────

/// DBSCAN: density-based clustering.
///
/// - `distances`: Pre-computed NxN symmetric distance matrix
/// - `eps`: Maximum distance for neighborhood membership
/// - `min_points`: Minimum neighborhood size to form a core point
///
/// Returns cluster assignments. Noise points get cluster = -1.
pub fn dbscan(distances: &[Vec<f64>], eps: f64, min_points: usize) -> Vec<ClusterAssignment> {
    let n = distances.len();
    // Build neighbor lists
    let neighbors: Vec<Vec<usize>> = (0..n)
        .map(|i| {
            (0..n)
                .filter(|&j| j != i && distances[i][j] <= eps)
                .collect()
        })
        .collect();

    let mut labels: Vec<i64> = vec![-2; n]; // -2 = unvisited, -1 = noise
    let mut cluster_id: i64 = 0;

    for i in 0..n {
        if labels[i] != -2 {
            continue; // already visited
        }
        if neighbors[i].len() < min_points {
            labels[i] = -1; // noise
            continue;
        }
        // Expand cluster from core point i
        labels[i] = cluster_id;
        let mut queue: Vec<usize> = neighbors[i].clone();
        let mut qi = 0;
        while qi < queue.len() {
            let q = queue[qi];
            qi += 1;
            if labels[q] == -1 {
                labels[q] = cluster_id; // border point
            }
            if labels[q] != -2 {
                continue; // already processed
            }
            labels[q] = cluster_id;
            if neighbors[q].len() >= min_points {
                // q is also a core point — expand
                for &nb in &neighbors[q] {
                    if (labels[nb] == -2 || labels[nb] == -1) && !queue.contains(&nb) {
                        queue.push(nb);
                    }
                }
            }
        }
        cluster_id += 1;
    }

    (0..n)
        .map(|i| ClusterAssignment {
            index: i,
            cluster: labels[i],
        })
        .collect()
}

// ── K-means ────────────────────────────────────────────────────────────────

/// K-means clustering on feature vectors.
///
/// Uses k-means++ initialization for better convergence.
/// Deterministic: uses a simple seeded selection based on input size.
///
/// - `features`: NxD feature matrix
/// - `k`: Number of clusters
/// - `max_iterations`: Maximum iteration count
pub fn kmeans(features: &[Vec<f64>], k: usize, max_iterations: usize) -> Vec<ClusterAssignment> {
    let n = features.len();
    if n == 0 || k == 0 {
        return Vec::new();
    }
    let k = k.min(n); // can't have more clusters than points
    let dims = features[0].len();

    // K-means++ initialization
    let mut centroids: Vec<Vec<f64>> = Vec::with_capacity(k);

    // First centroid: pick the point closest to the overall mean (deterministic)
    let mut mean = vec![0.0; dims];
    for f in features.iter() {
        for d in 0..dims {
            mean[d] += f[d];
        }
    }
    for m in mean.iter_mut() {
        *m /= n as f64;
    }
    let first = (0..n)
        .min_by(|&a, &b| {
            euclidean_distance(&features[a], &mean)
                .partial_cmp(&euclidean_distance(&features[b], &mean))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or(0);
    centroids.push(features[first].clone());

    // Remaining centroids: pick farthest point from nearest existing centroid (deterministic)
    for _ in 1..k {
        let mut best_idx = 0;
        let mut best_dist = f64::NEG_INFINITY;
        for (i, feat) in features.iter().enumerate() {
            let min_dist = centroids
                .iter()
                .map(|c| euclidean_distance(feat, c))
                .fold(f64::INFINITY, f64::min);
            if min_dist > best_dist {
                best_dist = min_dist;
                best_idx = i;
            }
        }
        centroids.push(features[best_idx].clone());
    }

    // Iterate: assign + recompute
    let mut assignments: Vec<usize> = vec![0; n];
    for _ in 0..max_iterations {
        let mut changed = false;

        // Assign each point to nearest centroid
        for i in 0..n {
            let nearest = (0..k)
                .min_by(|&a, &b| {
                    euclidean_distance(&features[i], &centroids[a])
                        .partial_cmp(&euclidean_distance(&features[i], &centroids[b]))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or(0);
            if assignments[i] != nearest {
                assignments[i] = nearest;
                changed = true;
            }
        }

        if !changed {
            break;
        }

        // Recompute centroids
        let mut new_centroids = vec![vec![0.0; dims]; k];
        let mut counts = vec![0usize; k];
        for i in 0..n {
            let c = assignments[i];
            counts[c] += 1;
            for d in 0..dims {
                new_centroids[c][d] += features[i][d];
            }
        }
        for c in 0..k {
            if counts[c] > 0 {
                for val in new_centroids[c].iter_mut() {
                    *val /= counts[c] as f64;
                }
            } else {
                // Empty cluster: keep previous centroid
                new_centroids[c] = centroids[c].clone();
            }
        }
        centroids = new_centroids;
    }

    (0..n)
        .map(|i| ClusterAssignment {
            index: i,
            cluster: assignments[i] as i64,
        })
        .collect()
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dbscan_two_clusters() {
        // Cluster 1: (0,0), (1,0), (0,1)
        // Cluster 2: (10,10), (11,10), (10,11)
        // Noise: (50,50)
        let features = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![10.0, 10.0],
            vec![11.0, 10.0],
            vec![10.0, 11.0],
            vec![50.0, 50.0],
        ];
        let dm = euclidean_distance_matrix(&features);
        let result = dbscan(&dm, 2.0, 2);

        // Points 0,1,2 should be in one cluster
        assert_eq!(result[0].cluster, result[1].cluster);
        assert_eq!(result[0].cluster, result[2].cluster);
        // Points 3,4,5 should be in another cluster
        assert_eq!(result[3].cluster, result[4].cluster);
        assert_eq!(result[3].cluster, result[5].cluster);
        // The two clusters should be different
        assert_ne!(result[0].cluster, result[3].cluster);
        // Point 6 should be noise
        assert_eq!(result[6].cluster, -1);
    }

    #[test]
    fn test_dbscan_all_one_cluster() {
        let features = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.5, 0.5]];
        let dm = euclidean_distance_matrix(&features);
        let result = dbscan(&dm, 2.0, 2);
        assert_eq!(result[0].cluster, result[1].cluster);
        assert_eq!(result[0].cluster, result[2].cluster);
        assert!(result[0].cluster >= 0);
    }

    #[test]
    fn test_dbscan_all_noise() {
        let features = vec![vec![0.0, 0.0], vec![100.0, 100.0], vec![200.0, 200.0]];
        let dm = euclidean_distance_matrix(&features);
        let result = dbscan(&dm, 1.0, 2);
        for r in &result {
            assert_eq!(r.cluster, -1);
        }
    }

    #[test]
    fn test_kmeans_two_clusters() {
        let features = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![10.0, 10.0],
            vec![11.0, 10.0],
            vec![10.0, 11.0],
        ];
        let result = kmeans(&features, 2, 100);
        // Points 0,1,2 should be in one cluster
        assert_eq!(result[0].cluster, result[1].cluster);
        assert_eq!(result[0].cluster, result[2].cluster);
        // Points 3,4,5 should be in another
        assert_eq!(result[3].cluster, result[4].cluster);
        assert_eq!(result[3].cluster, result[5].cluster);
        // Different clusters
        assert_ne!(result[0].cluster, result[3].cluster);
    }

    #[test]
    fn test_normalize_features() {
        let mut features = vec![vec![0.0, 100.0], vec![10.0, 200.0], vec![5.0, 150.0]];
        normalize_features(&mut features);
        assert_eq!(features[0], vec![0.0, 0.0]);
        assert_eq!(features[1], vec![1.0, 1.0]);
        assert_eq!(features[2], vec![0.5, 0.5]);
    }

    #[test]
    fn test_euclidean_distance_matrix() {
        let features = vec![vec![0.0, 0.0], vec![3.0, 4.0]];
        let dm = euclidean_distance_matrix(&features);
        assert!((dm[0][1] - 5.0).abs() < 1e-10);
        assert!((dm[1][0] - 5.0).abs() < 1e-10);
        assert_eq!(dm[0][0], 0.0);
    }

    #[test]
    fn test_haversine_distance_matrix() {
        // Oslo (59.91, 10.75) to Bergen (60.39, 5.32)
        let points = vec![(59.91, 10.75), (60.39, 5.32)];
        let dm = haversine_distance_matrix(&points);
        // Should be roughly 300km
        assert!(dm[0][1] > 250_000.0 && dm[0][1] < 350_000.0);
        assert_eq!(dm[0][0], 0.0);
    }
}
