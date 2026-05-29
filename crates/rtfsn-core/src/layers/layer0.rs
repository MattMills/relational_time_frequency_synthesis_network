use serde::{Deserialize, Serialize};

use crate::crypto::signatures::SignedMessage;
use crate::sync::clock::TimeExchange;
use crate::sync::geometry::CoordinateState;
use crate::types::{Epoch, NodeId};

/// Layer 0: Identity & raw measurement DHT.
///
/// This is the innermost, most private layer. Each node stores its own
/// encrypted measurement records here. Nothing leaves this layer without
/// passing through the L0→L1 valve (identity blinding).

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementRecord {
    pub peer: NodeId,
    pub exchange: TimeExchange,
    pub measured_rtt: f64,
    pub local_time: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalSolve {
    pub coordinates: CoordinateState,
    pub clock_offset: f64,
    pub clock_drift: f64,
    pub uncertainty: f64,
    pub peer_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Layer0Entry {
    pub node_id: NodeId,
    pub epoch: Epoch,
    pub measurements: Vec<MeasurementRecord>,
    pub solve: LocalSolve,
}

/// The local Layer 0 store — each node manages its own.
#[derive(Debug)]
pub struct Layer0Store {
    pub node_id: NodeId,
    entries: Vec<Layer0Entry>,
}

impl Layer0Store {
    pub fn new(node_id: NodeId) -> Self {
        Self {
            node_id,
            entries: Vec::new(),
        }
    }

    pub fn record_epoch(
        &mut self,
        epoch: Epoch,
        measurements: Vec<MeasurementRecord>,
        solve: LocalSolve,
    ) {
        self.entries.push(Layer0Entry {
            node_id: self.node_id,
            epoch,
            measurements,
            solve,
        });
    }

    pub fn latest_solve(&self) -> Option<&LocalSolve> {
        self.entries.last().map(|e| &e.solve)
    }

    pub fn get_epoch(&self, epoch: Epoch) -> Option<&Layer0Entry> {
        self.entries.iter().find(|e| e.epoch == epoch)
    }

    /// Serialize the entry for signing. The signed form can be stored
    /// in the DHT for audit purposes (encrypted to self).
    pub fn sign_entry(
        &self,
        epoch: Epoch,
        keypair: &crate::crypto::signatures::NodeKeypair,
    ) -> Option<SignedMessage> {
        let entry = self.get_epoch(epoch)?;
        let payload = serde_json::to_vec(entry).ok()?;
        Some(keypair.sign(&payload))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sync::clock::TimeExchange;
    use crate::sync::geometry::CoordinateState;

    #[test]
    fn test_layer0_store() {
        let node_id = NodeId([1u8; 32]);
        let mut store = Layer0Store::new(node_id);

        let measurements = vec![MeasurementRecord {
            peer: NodeId([2u8; 32]),
            exchange: TimeExchange::new(0.0, 0.025, 0.025, 0.050),
            measured_rtt: 0.050,
            local_time: 100.0,
        }];

        let solve = LocalSolve {
            coordinates: CoordinateState::new(),
            clock_offset: 0.001,
            clock_drift: 0.0,
            uncertainty: 0.005,
            peer_count: 1,
        };

        store.record_epoch(Epoch(1), measurements, solve);
        assert!(store.latest_solve().is_some());
        assert!(store.get_epoch(Epoch(1)).is_some());
        assert!(store.get_epoch(Epoch(2)).is_none());
    }
}
