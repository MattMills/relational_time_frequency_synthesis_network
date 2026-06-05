use curve25519_dalek::scalar::Scalar;
use serde::{Deserialize, Serialize};

use crate::crypto::pedersen::{PedersenCommitment, SerializableCommitment};
use crate::types::{Coordinates, Epoch};

use super::model::{NetworkGeoid, RegionId};

/// A node's position in the geoid hierarchy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeoidCoordinate {
    /// Region path from finest to coarsest level.
    pub region_path: Vec<RegionId>,
    /// The node's actual coordinate within its finest region.
    pub intra_region_position: Coordinates,
    pub epoch: Epoch,
}

impl GeoidCoordinate {
    /// Derive a geoid coordinate from Vivaldi coordinates and the current geoid.
    pub fn from_vivaldi(coords: &Coordinates, geoid: &NetworkGeoid) -> Self {
        let region_path: Vec<RegionId> = geoid
            .layers
            .iter()
            .filter_map(|layer| layer.assign_region(coords))
            .collect();

        Self {
            region_path,
            intra_region_position: coords.clone(),
            epoch: geoid.epoch,
        }
    }

    /// Number of levels this coordinate spans.
    pub fn depth(&self) -> usize {
        self.region_path.len()
    }

    /// Check whether two coordinates are in the same region at a given level.
    pub fn same_region_at_level(&self, other: &GeoidCoordinate, level: usize) -> bool {
        match (self.region_path.get(level), other.region_path.get(level)) {
            (Some(a), Some(b)) => a == b,
            _ => false,
        }
    }
}

/// Committed version: reveals region membership without revealing exact position.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommittedGeoidCoordinate {
    /// One Pedersen commitment per level in the region path.
    pub region_commitments: Vec<SerializableCommitment>,
    pub epoch: Epoch,
}

impl CommittedGeoidCoordinate {
    /// Commit each region ID in the path using Pedersen commitments.
    ///
    /// Returns (commitment, blindings_bytes) where blindings_bytes contains the
    /// 32-byte blinding factor for each region level.
    pub fn from_coordinate(coord: &GeoidCoordinate) -> (Self, Vec<[u8; 32]>) {
        let mut commitments = Vec::new();
        let mut blindings = Vec::new();

        for region_id in &coord.region_path {
            let mut seed = [0u8; 64];
            rand::fill(&mut seed);
            let blinding = Scalar::from_bytes_mod_order_wide(&seed);

            // Represent the RegionId as a scalar: take first 32 bytes mod order
            let region_scalar = Scalar::from_bytes_mod_order(region_id.0);
            let commitment = PedersenCommitment::commit(&region_scalar, &blinding);

            commitments.push(SerializableCommitment::from(&commitment));

            // Store just 32 bytes of blinding (truncated from 64-byte seed)
            let mut blinding_bytes = [0u8; 32];
            blinding_bytes.copy_from_slice(&seed[..32]);
            blindings.push(blinding_bytes);
        }

        (
            CommittedGeoidCoordinate {
                region_commitments: commitments,
                epoch: coord.epoch,
            },
            blindings,
        )
    }
}
