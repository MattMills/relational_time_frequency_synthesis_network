use serde::{Deserialize, Serialize};

use crate::crypto::pedersen::SerializableCommitment;
use crate::types::{Coordinates, Epoch};

/// Identifies a geoid region uniquely by a blake3 hash of its centroid, level, and epoch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RegionId(pub [u8; 32]);

impl RegionId {
    /// Create a region ID from a centroid, level, and epoch using blake3.
    pub fn from_centroid(centroid: &Coordinates, level: u8, epoch: Epoch) -> Self {
        let mut hasher = blake3::Hasher::new();
        for d in &centroid.dims {
            hasher.update(&d.to_le_bytes());
        }
        hasher.update(&[level]);
        hasher.update(&epoch.0.to_le_bytes());
        Self(*hasher.finalize().as_bytes())
    }

    /// Create a region ID from merging a set of region IDs.
    pub fn from_merge(ids: &[RegionId]) -> Self {
        let mut hasher = blake3::Hasher::new();
        for id in ids {
            hasher.update(&id.0);
        }
        Self(*hasher.finalize().as_bytes())
    }
}

/// Privacy-preserving committed latency profile using Pedersen commitments.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommittedLatencyProfile {
    pub committed_mean: SerializableCommitment,
    pub committed_variance: SerializableCommitment,
    pub sample_count: u32,
}

/// An edge between two geoid regions, recording latency profile.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeoidEdge {
    pub from: RegionId,
    pub to: RegionId,
    pub profile: CommittedLatencyProfile,
    /// Plaintext mean latency in nanoseconds (for routing decisions).
    pub mean_nanos_plaintext: i64,
}

/// A single region in the geoid hierarchy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeoidRegion {
    pub id: RegionId,
    pub centroid: Coordinates,
    pub radius: f64,
    pub member_count: u32,
    pub intra_latency: CommittedLatencyProfile,
    /// IDs of child regions (at the finer level).
    pub child_region_ids: Vec<RegionId>,
    /// Edges to neighboring regions at the same level.
    pub inter_edges: Vec<GeoidEdge>,
    /// Residual between observed and predicted timing geometry (seconds).
    pub geoid_residual: f64,
}

/// A single level of the geoid hierarchy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeoidLayer {
    pub level: u8,
    pub resolution_nanos: u64,
    pub regions: Vec<GeoidRegion>,
    pub epoch: Epoch,
}

impl GeoidLayer {
    /// Find a region by its ID.
    pub fn find_region(&self, id: &RegionId) -> Option<&GeoidRegion> {
        self.regions.iter().find(|r| &r.id == id)
    }

    /// Assign coordinates to the nearest region centroid.
    pub fn assign_region(&self, coords: &Coordinates) -> Option<RegionId> {
        self.regions
            .iter()
            .min_by(|a, b| {
                let da = a.centroid.distance(coords);
                let db = b.centroid.distance(coords);
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|r| r.id)
    }
}

/// The full hierarchical geoid model of the network's timing topology.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkGeoid {
    /// Layers ordered finest to coarsest (index 0 = finest).
    pub layers: Vec<GeoidLayer>,
    pub epoch: Epoch,
    pub refinement_generation: u64,
    pub prev_hash: [u8; 32],
}

impl NetworkGeoid {
    /// Create a new empty geoid for the given epoch.
    pub fn new(epoch: Epoch) -> Self {
        Self {
            layers: Vec::new(),
            epoch,
            refinement_generation: 0,
            prev_hash: [0u8; 32],
        }
    }

    /// Number of levels in the hierarchy.
    pub fn depth(&self) -> u8 {
        self.layers.len() as u8
    }

    /// For each level, return which region the coordinates fall in.
    pub fn locate(&self, coords: &Coordinates) -> Vec<Option<RegionId>> {
        self.layers
            .iter()
            .map(|layer| layer.assign_region(coords))
            .collect()
    }

    /// Compute a stable hash of this geoid (epoch + all region IDs + member counts).
    pub fn hash(&self) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&self.epoch.0.to_le_bytes());
        for layer in &self.layers {
            for region in &layer.regions {
                hasher.update(&region.id.0);
                hasher.update(&region.member_count.to_le_bytes());
            }
        }
        *hasher.finalize().as_bytes()
    }

    /// Compute a normalized [0, 1] change metric compared to a previous geoid.
    pub fn delta_from(&self, prev: &NetworkGeoid) -> f64 {
        if self.layers.len() != prev.layers.len() {
            return 1.0;
        }
        if self.layers.is_empty() {
            return 0.0;
        }

        let mut total_changed = 0usize;
        let mut total_regions = 0usize;

        for (new_layer, old_layer) in self.layers.iter().zip(prev.layers.iter()) {
            total_regions += new_layer.regions.len().max(old_layer.regions.len());
            // Count regions that appear in new but not old, or changed member count
            for new_region in &new_layer.regions {
                match old_layer.find_region(&new_region.id) {
                    Some(old_region) => {
                        if old_region.member_count != new_region.member_count {
                            total_changed += 1;
                        }
                    }
                    None => {
                        total_changed += 1;
                    }
                }
            }
            // Regions that disappeared
            for old_region in &old_layer.regions {
                if new_layer.find_region(&old_region.id).is_none() {
                    total_changed += 1;
                }
            }
        }

        if total_regions == 0 {
            return 0.0;
        }

        (total_changed as f64 / total_regions as f64).min(1.0)
    }
}
