use serde::{Deserialize, Serialize};

use crate::crypto::blinding::{BlindedIdentity, BlindingProof};
use crate::sync::geometry::CoordinateState;
use crate::types::Epoch;

/// Layer 1: Geometric embedding DHT.
///
/// Nodes here are pseudonymous per epoch. Each publishes its solved
/// coordinates and clock estimate, along with ZK proofs of validity.
/// The blinded identity cannot be linked across epochs or traced to Layer 0.

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Layer1Beacon {
    pub blinded_id: BlindedIdentity,
    pub epoch: Epoch,
    pub coordinates: CoordinateState,
    pub clock_offset: f64,
    pub clock_drift: f64,
    pub uncertainty: f64,
    pub peer_count: u32,
    pub consistency_score: f64,

    pub proof_of_identity: BlindingProof,
    pub proof_of_peer_count: Option<PeerCountProof>,
}

/// Proof that the node incorporated at least k distinct peers.
/// In production this would be a ZK-SNARK; here we use a simplified
/// commitment-based scheme.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerCountProof {
    pub claimed_count: u32,
    pub merkle_root: [u8; 32],
}

impl Layer1Beacon {
    pub fn verify_identity(&self) -> bool {
        self.proof_of_identity.verify()
    }

    pub fn key(&self) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&self.blinded_id.commitment);
        hasher.update(&self.epoch.0.to_le_bytes());
        *hasher.finalize().as_bytes()
    }
}

/// The Layer 1 DHT stores beacons indexed by blinded identity + epoch.
#[derive(Debug)]
pub struct Layer1Store {
    beacons: Vec<Layer1Beacon>,
}

impl Layer1Store {
    pub fn new() -> Self {
        Self {
            beacons: Vec::new(),
        }
    }

    pub fn insert(&mut self, beacon: Layer1Beacon) -> bool {
        if !beacon.verify_identity() {
            return false;
        }

        let key = beacon.key();
        if self.beacons.iter().any(|b| b.key() == key) {
            return false;
        }

        self.beacons.push(beacon);
        true
    }

    pub fn get_epoch(&self, epoch: Epoch) -> Vec<&Layer1Beacon> {
        self.beacons.iter().filter(|b| b.epoch == epoch).collect()
    }

    pub fn peer_coordinates(
        &self,
        epoch: Epoch,
    ) -> Vec<(&CoordinateState, f64)> {
        self.get_epoch(epoch)
            .into_iter()
            .map(|b| (&b.coordinates, b.clock_offset))
            .collect()
    }
}

impl Default for Layer1Store {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto::blinding::BlindingSecret;
    use crate::types::NodeId;

    #[test]
    fn test_layer1_beacon_roundtrip() {
        let node_id = NodeId([42u8; 32]);
        let epoch = Epoch(1);

        let secret = BlindingSecret::new(&node_id, epoch);
        let blinded_id = secret.blind();
        let proof = secret.prove_knowledge();

        let beacon = Layer1Beacon {
            blinded_id,
            epoch,
            coordinates: CoordinateState::new(),
            clock_offset: 0.001,
            clock_drift: 0.0,
            uncertainty: 0.005,
            peer_count: 5,
            consistency_score: 0.95,
            proof_of_identity: proof,
            proof_of_peer_count: None,
        };

        assert!(beacon.verify_identity());

        let mut store = Layer1Store::new();
        assert!(store.insert(beacon));
        assert_eq!(store.get_epoch(Epoch(1)).len(), 1);
    }
}
