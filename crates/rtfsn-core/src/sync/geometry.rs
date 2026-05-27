use crate::types::{Coordinates, NodeId, COORDINATE_DIMENSIONS};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Vivaldi-style network coordinate embedding.
///
/// Each node maintains coordinates in a d-dimensional space such that
/// the Euclidean distance between two nodes' coordinates approximates
/// the measured RTT between them.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinateState {
    pub position: Coordinates,
    pub error: f64,
}

impl CoordinateState {
    pub fn new() -> Self {
        let mut dims = [0.0; COORDINATE_DIMENSIONS];
        let mut bytes = [0u8; COORDINATE_DIMENSIONS * 4];
        rand::fill(&mut bytes);
        for i in 0..COORDINATE_DIMENSIONS {
            let b: [u8; 4] = bytes[i * 4..(i + 1) * 4].try_into().unwrap();
            let val = u32::from_le_bytes(b) as f64 / u32::MAX as f64;
            dims[i] = (val - 0.5) * 0.001;
        }

        Self {
            position: Coordinates { dims },
            error: 1.0,
        }
    }

    pub fn predicted_rtt(&self, other: &CoordinateState) -> f64 {
        self.position.distance(&other.position)
    }

    /// Update coordinates given a measured RTT to a peer.
    pub fn update(
        &mut self,
        peer: &CoordinateState,
        measured_rtt: f64,
    ) -> f64 {
        let dist = self.position.distance(&peer.position);
        let error = measured_rtt - dist;
        let relative_error = if measured_rtt > 1e-12 {
            error.abs() / measured_rtt
        } else {
            error.abs()
        };

        let weight = if self.error + peer.error > 1e-12 {
            self.error / (self.error + peer.error)
        } else {
            0.5
        };

        let ce = 0.25;
        self.error = relative_error * ce * weight
            + self.error * (1.0 - ce * weight);

        let delta = 0.25 * weight;

        // Unit vector from peer toward self (away from peer)
        let away = self.position.subtract(&peer.position);
        let dir_mag = away.magnitude();

        if dir_mag > 1e-12 {
            let unit = away.normalize();
            // Positive error = too close, move away; negative = too far, move toward
            self.position = self.position.add_scaled(&unit, error * delta);
        } else if error.abs() > 1e-12 {
            let mut nudge_dims = [0.0; COORDINATE_DIMENSIONS];
            nudge_dims[0] = error.signum() * measured_rtt * delta;
            self.position = self.position.add_scaled(
                &Coordinates { dims: nudge_dims },
                1.0,
            );
        }

        relative_error
    }
}

impl Default for CoordinateState {
    fn default() -> Self {
        Self::new()
    }
}

/// Manages the geometric embedding for the local node and its peer observations.
#[derive(Debug)]
pub struct GeometryEngine {
    pub local: CoordinateState,
    pub peers: HashMap<NodeId, CoordinateState>,
}

impl GeometryEngine {
    pub fn new() -> Self {
        Self {
            local: CoordinateState::new(),
            peers: HashMap::new(),
        }
    }

    pub fn observe_peer(
        &mut self,
        peer_id: NodeId,
        peer_coords: CoordinateState,
        measured_rtt: f64,
    ) -> f64 {
        let error = self.local.update(&peer_coords, measured_rtt);
        self.peers.insert(peer_id, peer_coords);
        error
    }

    pub fn predicted_rtt_to(&self, peer_id: &NodeId) -> Option<f64> {
        self.peers
            .get(peer_id)
            .map(|p| self.local.predicted_rtt(p))
    }

    pub fn residual(
        &self,
        peer_id: &NodeId,
        measured_rtt: f64,
    ) -> Option<f64> {
        self.predicted_rtt_to(peer_id)
            .map(|predicted| (measured_rtt - predicted).abs())
    }
}

impl Default for GeometryEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_node_at(dims: [f64; COORDINATE_DIMENSIONS]) -> CoordinateState {
        CoordinateState {
            position: Coordinates { dims },
            error: 1.0,
        }
    }

    #[test]
    fn test_vivaldi_convergence() {
        let true_distance = 0.05;

        let mut a = CoordinateState::new();
        let mut b = make_node_at({
            let mut d = [0.0; COORDINATE_DIMENSIONS];
            d[0] = true_distance;
            d
        });

        for _ in 0..500 {
            let b_snap = b.clone();
            a.update(&b_snap, true_distance);
            let a_snap = a.clone();
            b.update(&a_snap, true_distance);
        }

        let predicted = a.predicted_rtt(&b);
        let error = (predicted - true_distance).abs();
        assert!(
            error < 0.01,
            "predicted={predicted}, true={true_distance}, error={error}"
        );
    }

    #[test]
    fn test_triangle_consistency() {
        let rtts = [
            (0, 1, 0.030),
            (0, 2, 0.050),
            (1, 2, 0.040),
        ];

        let mut nodes = vec![
            CoordinateState::new(),
            CoordinateState::new(),
            CoordinateState::new(),
        ];

        for _ in 0..1000 {
            for &(i, j, rtt) in &rtts {
                let peer_clone = nodes[j].clone();
                nodes[i].update(&peer_clone, rtt);
                let peer_clone = nodes[i].clone();
                nodes[j].update(&peer_clone, rtt);
            }
        }

        for &(i, j, rtt) in &rtts {
            let predicted = nodes[i].predicted_rtt(&nodes[j]);
            let error = (predicted - rtt).abs();
            assert!(
                error < 0.02,
                "pair ({i},{j}): predicted={predicted}, true={rtt}, error={error}"
            );
        }
    }
}
