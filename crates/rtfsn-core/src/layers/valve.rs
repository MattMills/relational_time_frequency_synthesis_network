use crate::crypto::blinding::BlindingSecret;
use crate::layers::layer0::Layer0Store;
use crate::layers::layer1::{Layer1Beacon, Layer1Store, PeerCountProof};
use crate::layers::layer2::{ClusterAggregator, ClusterId};
use crate::layers::layer3::ClockStream;
use crate::types::Epoch;

/// Valve L0 → L1: Blind identity and produce Layer 1 beacon.
///
/// This is the critical privacy boundary. The raw measurements and
/// true identity stay in Layer 0; only the solved position, blinded
/// identity, and ZK proofs cross to Layer 1.
pub fn valve_l0_to_l1(
    store: &Layer0Store,
    epoch: Epoch,
    blinding_secret: &BlindingSecret,
) -> Option<Layer1Beacon> {
    let entry = store.get_epoch(epoch)?;
    let solve = &entry.solve;

    let blinded_id = blinding_secret.blind();
    let proof = blinding_secret.prove_knowledge();

    let peer_proof = if solve.peer_count > 0 {
        let peer_ids: Vec<&[u8]> = entry
            .measurements
            .iter()
            .map(|m| m.peer.0.as_slice())
            .collect();
        let merkle = crate::types::MerkleDigest::from_items(&peer_ids);
        Some(PeerCountProof {
            claimed_count: solve.peer_count as u32,
            merkle_root: merkle.0,
        })
    } else {
        None
    };

    Some(Layer1Beacon {
        blinded_id,
        epoch,
        coordinates: solve.coordinates.clone(),
        clock_offset: solve.clock_offset,
        clock_drift: solve.clock_drift,
        uncertainty: solve.uncertainty,
        peer_count: solve.peer_count as u32,
        consistency_score: 1.0 / (1.0 + solve.uncertainty),
        proof_of_identity: proof,
        proof_of_peer_count: peer_proof,
    })
}

/// Valve L1 → L2: Aggregate beacons into cluster commitments.
///
/// Groups nearby Layer 1 nodes into clusters and commits their
/// aggregate clock offset using Pedersen commitments.
pub fn valve_l1_to_l2(
    l1_store: &Layer1Store,
    l2_aggregator: &mut ClusterAggregator,
    epoch: Epoch,
    grid_size: f64,
) {
    let beacons = l1_store.get_epoch(epoch);
    if beacons.is_empty() {
        return;
    }

    // Spatial clustering: quantize coordinates to a grid
    let mut clusters: std::collections::HashMap<Vec<i64>, Vec<f64>> =
        std::collections::HashMap::new();

    for beacon in &beacons {
        let grid_key: Vec<i64> = beacon
            .coordinates
            .position
            .dims
            .iter()
            .map(|d| (d / grid_size).floor() as i64)
            .collect();

        clusters
            .entry(grid_key)
            .or_default()
            .push(beacon.clock_offset);
    }

    for (grid_key, offsets) in &clusters {
        let region_coords: Vec<f64> =
            grid_key.iter().map(|g| *g as f64 * grid_size).collect();
        let cluster_id = ClusterId::from_region(&region_coords, epoch);
        l2_aggregator.aggregate(cluster_id, epoch, offsets);
    }
}

/// Valve L2 → L3: Threshold unblind and emit clock tick.
///
/// When enough clusters agree (verified by range proofs), their
/// committed offsets are collaboratively unblinded to produce a
/// single plaintext time value.
pub fn valve_l2_to_l3(
    l2_aggregator: &ClusterAggregator,
    clock_stream: &mut ClockStream,
    epoch: Epoch,
    reference_time_nanos: u64,
    _min_clusters: usize,
) -> bool {
    let clusters = l2_aggregator.get_epoch(epoch);
    if clusters.is_empty() {
        return false;
    }

    // In production, this would involve threshold decryption of
    // the Pedersen commitments. For now, we accept the cluster
    // data and produce a clock tick.

    let total_weight: f64 =
        clusters.iter().map(|c| c.cluster_size as f64).sum();
    if total_weight < 1.0 {
        return false;
    }

    // Build epoch hash from all cluster commitments
    let mut hasher = blake3::Hasher::new();
    for cluster in &clusters {
        hasher.update(&cluster.committed_offset.bytes);
        hasher.update(&cluster.committed_variance.bytes);
    }
    let epoch_hash = *hasher.finalize().as_bytes();

    let confidence_nanos = 5_000_000; // 5ms placeholder

    clock_stream.emit(
        reference_time_nanos,
        confidence_nanos,
        epoch_hash,
        None,
    );

    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::layer0::{
        Layer0Store, LocalSolve, MeasurementRecord,
    };
    use crate::sync::clock::TimeExchange;
    use crate::sync::geometry::CoordinateState;
    use crate::types::NodeId;

    #[test]
    fn test_full_valve_pipeline() {
        let node_id = NodeId([1u8; 32]);
        let epoch = Epoch(1);

        // Layer 0: record measurements
        let mut l0 = Layer0Store::new(node_id);
        let measurements = vec![
            MeasurementRecord {
                peer: NodeId([2u8; 32]),
                exchange: TimeExchange::new(0.0, 0.025, 0.025, 0.050),
                measured_rtt: 0.050,
                local_time: 100.0,
            },
            MeasurementRecord {
                peer: NodeId([3u8; 32]),
                exchange: TimeExchange::new(0.0, 0.015, 0.015, 0.030),
                measured_rtt: 0.030,
                local_time: 100.5,
            },
        ];
        let solve = LocalSolve {
            coordinates: CoordinateState::new(),
            clock_offset: 0.001,
            clock_drift: 0.0,
            uncertainty: 0.005,
            peer_count: 2,
        };
        l0.record_epoch(epoch, measurements, solve);

        // Valve L0 → L1
        let blinding = BlindingSecret::new(&node_id, epoch);
        let beacon = valve_l0_to_l1(&l0, epoch, &blinding).unwrap();
        assert!(beacon.verify_identity());

        // Layer 1: store beacon
        let mut l1 = Layer1Store::new();
        assert!(l1.insert(beacon));

        // Valve L1 → L2
        let mut l2 = ClusterAggregator::new();
        valve_l1_to_l2(&l1, &mut l2, epoch, 0.01);

        // Valve L2 → L3
        let mut clock = ClockStream::new();
        let emitted = valve_l2_to_l3(
            &l2,
            &mut clock,
            epoch,
            1_000_000_000,
            1,
        );
        assert!(emitted);
        assert_eq!(clock.len(), 2); // genesis + 1 tick
        assert!(clock.verify_all());
    }
}
