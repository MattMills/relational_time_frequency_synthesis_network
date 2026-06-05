use curve25519_dalek::scalar::Scalar;

use crate::crypto::pedersen::{PedersenCommitment, SerializableCommitment};
use crate::layers::layer1::Layer1Beacon;
use crate::types::{Coordinates, Epoch, COORDINATE_DIMENSIONS};

use super::model::{
    CommittedLatencyProfile, GeoidEdge, GeoidLayer, GeoidRegion, NetworkGeoid, RegionId,
};

/// Builds a hierarchical NetworkGeoid from Layer1 beacons by iteratively coarsening.
pub struct CoarseningOperator {
    /// Finest resolution in seconds (default: 2ms).
    pub base_resolution_secs: f64,
    /// Each successive level is this many times coarser (default: 4x).
    pub scale_factor: f64,
    /// Number of levels in the hierarchy (default: 4).
    pub levels: u8,
}

impl CoarseningOperator {
    pub fn new() -> Self {
        Self {
            base_resolution_secs: 0.002,
            scale_factor: 4.0,
            levels: 4,
        }
    }

    pub fn with_levels(levels: u8) -> Self {
        Self {
            levels,
            ..Self::new()
        }
    }

    /// Build the full hierarchy from a set of L1 beacons.
    pub fn build_hierarchy(
        &self,
        beacons: &[&Layer1Beacon],
        epoch: Epoch,
    ) -> NetworkGeoid {
        if beacons.is_empty() {
            return NetworkGeoid::new(epoch);
        }

        let mut layers = Vec::new();

        // Build finest layer from beacons
        let leaf = self.build_leaf_layer(beacons, epoch);
        layers.push(leaf);

        // Coarsen iteratively
        for _i in 1..self.levels {
            let prev = layers.last().unwrap();
            if prev.regions.is_empty() {
                break;
            }
            let coarser = self.coarsen_layer(prev, epoch);
            layers.push(coarser);
        }

        let hash = {
            let mut hasher = blake3::Hasher::new();
            hasher.update(&epoch.0.to_le_bytes());
            for layer in &layers {
                for r in &layer.regions {
                    hasher.update(&r.id.0);
                }
            }
            *hasher.finalize().as_bytes()
        };

        NetworkGeoid {
            layers,
            epoch,
            refinement_generation: 0,
            prev_hash: hash,
        }
    }

    /// Build the finest geoid layer directly from beacons.
    fn build_leaf_layer(
        &self,
        beacons: &[&Layer1Beacon],
        epoch: Epoch,
    ) -> GeoidLayer {
        let threshold = self.base_resolution_secs;
        let resolution_nanos = (threshold * 1e9) as u64;

        // Each entry: (running centroid (sum of dims), count, list of clock offsets)
        struct RegionAccum {
            sum: [f64; COORDINATE_DIMENSIONS],
            count: u32,
            offsets: Vec<f64>,
        }

        let mut accums: Vec<RegionAccum> = Vec::new();

        for beacon in beacons {
            let pos = &beacon.coordinates.position;
            // Find nearest existing accumulator within threshold
            let nearest = accums.iter_mut().enumerate().find(|(_, acc)| {
                let centroid = Coordinates {
                    dims: std::array::from_fn(|i| acc.sum[i] / acc.count as f64),
                };
                centroid.distance(pos) < threshold
            });

            match nearest {
                Some((_, acc)) => {
                    // Weighted incremental update
                    for i in 0..COORDINATE_DIMENSIONS {
                        acc.sum[i] += pos.dims[i];
                    }
                    acc.count += 1;
                    acc.offsets.push(beacon.clock_offset);
                }
                None => {
                    accums.push(RegionAccum {
                        sum: pos.dims,
                        count: 1,
                        offsets: vec![beacon.clock_offset],
                    });
                }
            }
        }

        // Build regions with stable IDs from final centroids
        let regions: Vec<GeoidRegion> = accums
            .iter()
            .enumerate()
            .map(|(_idx, acc)| {
                let centroid = Coordinates {
                    dims: std::array::from_fn(|i| acc.sum[i] / acc.count as f64),
                };
                let id = RegionId::from_centroid(&centroid, 0, epoch);

                // Compute intra-region latency profile from clock offsets
                let mean_nanos = if acc.offsets.is_empty() {
                    0i64
                } else {
                    let mean_secs =
                        acc.offsets.iter().sum::<f64>() / acc.offsets.len() as f64;
                    (mean_secs * 1e9) as i64
                };
                let variance_nanos = if acc.offsets.len() > 1 {
                    let mean_secs =
                        acc.offsets.iter().sum::<f64>() / acc.offsets.len() as f64;
                    let var =
                        acc.offsets.iter().map(|o| (o - mean_secs).powi(2)).sum::<f64>()
                            / acc.offsets.len() as f64;
                    (var * 1e18) as i64
                } else {
                    0
                };
                let profile =
                    self.commit_latency(mean_nanos, variance_nanos, acc.count);

                // Radius: half-threshold
                let radius = threshold / 2.0;

                // Geoid residual: spread of clock offsets as a proxy for timing error
                let residual = if acc.offsets.len() > 1 {
                    let mean_secs =
                        acc.offsets.iter().sum::<f64>() / acc.offsets.len() as f64;
                    acc.offsets.iter().map(|o| (o - mean_secs).abs()).sum::<f64>()
                        / acc.offsets.len() as f64
                } else {
                    0.0
                };

                GeoidRegion {
                    id,
                    centroid,
                    radius,
                    member_count: acc.count,
                    intra_latency: profile,
                    child_region_ids: Vec::new(),
                    inter_edges: Vec::new(), // filled below
                    geoid_residual: residual,
                }
            })
            .collect();

        // Build edges between nearby regions (within 3x threshold)
        let edge_threshold = threshold * 3.0;
        let regions = self.add_inter_edges(regions, edge_threshold);

        GeoidLayer {
            level: 0,
            resolution_nanos,
            regions,
            epoch,
        }
    }

    /// Coarsen a layer into the next (coarser) level.
    fn coarsen_layer(&self, layer: &GeoidLayer, epoch: Epoch) -> GeoidLayer {
        let next_level = layer.level + 1;
        let threshold =
            self.base_resolution_secs * self.scale_factor.powi(next_level as i32);
        let resolution_nanos = (threshold * 1e9) as u64;

        let mut merged: Vec<bool> = vec![false; layer.regions.len()];
        let mut super_regions: Vec<GeoidRegion> = Vec::new();

        for i in 0..layer.regions.len() {
            if merged[i] {
                continue;
            }
            // Collect all nearby un-merged regions
            let mut group_indices = vec![i];
            merged[i] = true;

            for j in (i + 1)..layer.regions.len() {
                if !merged[j] {
                    let dist = layer.regions[i]
                        .centroid
                        .distance(&layer.regions[j].centroid);
                    if dist < threshold {
                        group_indices.push(j);
                        merged[j] = true;
                    }
                }
            }

            let group: Vec<&GeoidRegion> =
                group_indices.iter().map(|&k| &layer.regions[k]).collect();
            let super_region = self.merge_regions(group, next_level, epoch);
            super_regions.push(super_region);
        }

        // Build inter-edges between super-regions
        let edge_threshold = threshold * 3.0;
        let super_regions = self.add_inter_edges(super_regions, edge_threshold);

        GeoidLayer {
            level: next_level,
            resolution_nanos,
            regions: super_regions,
            epoch,
        }
    }

    /// Merge a group of fine-level regions into one coarser super-region.
    fn merge_regions(
        &self,
        regions: Vec<&GeoidRegion>,
        level: u8,
        epoch: Epoch,
    ) -> GeoidRegion {
        let total_members: u32 = regions.iter().map(|r| r.member_count).sum();
        let weight_sum = total_members as f64;

        // Weighted centroid
        let mut centroid_dims = [0.0f64; COORDINATE_DIMENSIONS];
        for r in &regions {
            let w = r.member_count as f64 / weight_sum.max(1.0);
            for i in 0..COORDINATE_DIMENSIONS {
                centroid_dims[i] += r.centroid.dims[i] * w;
            }
        }
        let centroid = Coordinates { dims: centroid_dims };
        let id = RegionId::from_centroid(&centroid, level, epoch);

        // Radius: max distance from centroid to any child centroid
        let radius = regions
            .iter()
            .map(|r| r.centroid.distance(&centroid) + r.radius)
            .fold(0.0f64, f64::max);

        // Aggregate intra-latency: use 0 as mean (committed values are opaque).
        let mean_nanos = if total_members > 0 {
            0i64 // Committed values are opaque; plaintext mean unavailable here
        } else {
            0
        };

        let profile = self.commit_latency(mean_nanos, 0, total_members);

        // Collect child region IDs
        let child_region_ids: Vec<RegionId> = regions.iter().map(|r| r.id).collect();

        // Propagate worst residual from children
        let geoid_residual = regions
            .iter()
            .map(|r| r.geoid_residual)
            .fold(0.0f64, f64::max);

        GeoidRegion {
            id,
            centroid,
            radius,
            member_count: total_members,
            intra_latency: profile,
            child_region_ids,
            inter_edges: Vec::new(), // will be filled by add_inter_edges
            geoid_residual,
        }
    }

    /// Add inter-region edges to a set of regions for pairs within the threshold distance.
    fn add_inter_edges(
        &self,
        mut regions: Vec<GeoidRegion>,
        threshold: f64,
    ) -> Vec<GeoidRegion> {
        // Collect edge data first to avoid borrow issues
        let n = regions.len();
        let mut edges_to_add: Vec<(usize, GeoidEdge)> = Vec::new();

        for i in 0..n {
            for j in (i + 1)..n {
                let dist = regions[i].centroid.distance(&regions[j].centroid);
                if dist < threshold {
                    let mean_nanos = (dist * 1e9) as i64;
                    let profile = self.commit_latency(mean_nanos, 0, 1);
                    let edge_ij = GeoidEdge {
                        from: regions[i].id,
                        to: regions[j].id,
                        profile: profile.clone(),
                        mean_nanos_plaintext: mean_nanos,
                    };
                    let edge_ji = GeoidEdge {
                        from: regions[j].id,
                        to: regions[i].id,
                        profile,
                        mean_nanos_plaintext: mean_nanos,
                    };
                    edges_to_add.push((i, edge_ij));
                    edges_to_add.push((j, edge_ji));
                }
            }
        }

        for (idx, edge) in edges_to_add {
            regions[idx].inter_edges.push(edge);
        }

        regions
    }

    /// Create a CommittedLatencyProfile by Pedersen-committing the mean and variance.
    fn commit_latency(
        &self,
        mean_nanos: i64,
        variance_nanos: i64,
        count: u32,
    ) -> CommittedLatencyProfile {
        let mut blinding_bytes = [0u8; 64];
        rand::fill(&mut blinding_bytes);
        let blinding_mean = Scalar::from_bytes_mod_order_wide(&blinding_bytes);

        let mut var_blinding_bytes = [0u8; 64];
        rand::fill(&mut var_blinding_bytes);
        let blinding_var = Scalar::from_bytes_mod_order_wide(&var_blinding_bytes);

        let mean_scalar = if mean_nanos >= 0 {
            Scalar::from(mean_nanos as u64)
        } else {
            -Scalar::from((-mean_nanos) as u64)
        };
        let var_scalar = if variance_nanos >= 0 {
            Scalar::from(variance_nanos as u64)
        } else {
            -Scalar::from((-variance_nanos) as u64)
        };

        let committed_mean =
            SerializableCommitment::from(&PedersenCommitment::commit(&mean_scalar, &blinding_mean));
        let committed_variance = SerializableCommitment::from(&PedersenCommitment::commit(
            &var_scalar,
            &blinding_var,
        ));

        CommittedLatencyProfile {
            committed_mean,
            committed_variance,
            sample_count: count,
        }
    }
}

impl Default for CoarseningOperator {
    fn default() -> Self {
        Self::new()
    }
}
