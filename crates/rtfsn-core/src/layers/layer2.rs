use serde::{Deserialize, Serialize};

use crate::crypto::pedersen::SerializableCommitment;
use crate::types::Epoch;

/// Layer 2: Consensus clock DHT.
///
/// Stores cluster-level committed time estimates. Individual node data
/// is aggregated into cluster means using homomorphic commitments.
/// Inter-cluster range proofs demonstrate agreement without revealing values.

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ClusterId(pub [u8; 32]);

impl ClusterId {
    pub fn from_region(coords: &[f64], epoch: Epoch) -> Self {
        let mut hasher = blake3::Hasher::new();
        for c in coords {
            hasher.update(&c.to_le_bytes());
        }
        hasher.update(&epoch.0.to_le_bytes());
        Self(*hasher.finalize().as_bytes())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterAgreementProof {
    pub other_cluster: ClusterId,
    pub within_tolerance: bool,
    pub proof_hash: [u8; 32],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Layer2Entry {
    pub cluster_id: ClusterId,
    pub epoch: Epoch,
    pub committed_offset: SerializableCommitment,
    pub committed_variance: SerializableCommitment,
    pub cluster_size: u32,
    pub agreement_proofs: Vec<ClusterAgreementProof>,
}

/// Aggregates Layer 1 beacons into cluster-level commitments.
pub struct ClusterAggregator {
    entries: Vec<Layer2Entry>,
}

impl ClusterAggregator {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Aggregate a set of clock offsets into a cluster entry.
    /// The offsets are committed using Pedersen commitments so the
    /// actual values remain hidden.
    pub fn aggregate(
        &mut self,
        cluster_id: ClusterId,
        epoch: Epoch,
        offsets: &[f64],
    ) -> Option<&Layer2Entry> {
        if offsets.is_empty() {
            return None;
        }

        use crate::crypto::pedersen::PedersenCommitment;
        use curve25519_dalek::scalar::Scalar;

        let mean = offsets.iter().sum::<f64>() / offsets.len() as f64;
        let variance = offsets.iter().map(|o| (o - mean).powi(2)).sum::<f64>()
            / offsets.len() as f64;

        let mut blinding_bytes = [0u8; 64];
        rand::fill(&mut blinding_bytes);
        let blinding = Scalar::from_bytes_mod_order_wide(&blinding_bytes);

        let offset_commitment =
            PedersenCommitment::commit_f64(mean, &blinding);

        let mut var_blinding_bytes = [0u8; 64];
        rand::fill(&mut var_blinding_bytes);
        let var_blinding =
            Scalar::from_bytes_mod_order_wide(&var_blinding_bytes);

        let variance_commitment =
            PedersenCommitment::commit_f64(variance, &var_blinding);

        let entry = Layer2Entry {
            cluster_id,
            epoch,
            committed_offset: SerializableCommitment::from(
                &offset_commitment,
            ),
            committed_variance: SerializableCommitment::from(
                &variance_commitment,
            ),
            cluster_size: offsets.len() as u32,
            agreement_proofs: Vec::new(),
        };

        self.entries.push(entry);
        self.entries.last()
    }

    pub fn get_epoch(&self, epoch: Epoch) -> Vec<&Layer2Entry> {
        self.entries.iter().filter(|e| e.epoch == epoch).collect()
    }

    /// Check if enough clusters agree for the epoch to produce
    /// a Layer 3 output.
    pub fn has_consensus(&self, epoch: Epoch, min_clusters: usize) -> bool {
        let clusters = self.get_epoch(epoch);
        if clusters.len() < min_clusters {
            return false;
        }

        // Check that clusters have mutual agreement proofs
        let with_agreements: usize = clusters
            .iter()
            .filter(|c| !c.agreement_proofs.is_empty())
            .count();

        with_agreements >= min_clusters
    }
}

impl Default for ClusterAggregator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cluster_aggregation() {
        let mut agg = ClusterAggregator::new();
        let cluster_id = ClusterId::from_region(&[0.1, 0.2], Epoch(1));

        let offsets = vec![0.001, 0.0012, 0.0008, 0.0011, 0.0009];
        let entry = agg.aggregate(cluster_id, Epoch(1), &offsets);
        assert!(entry.is_some());

        let entry = entry.unwrap();
        assert_eq!(entry.cluster_size, 5);
    }
}
