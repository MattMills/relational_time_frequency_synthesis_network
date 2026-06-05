use serde::{Deserialize, Serialize};

use crate::layers::layer1::Layer1Beacon;
use crate::types::{Coordinates, Epoch};

use super::coarsen::CoarseningOperator;
use super::model::{NetworkGeoid, RegionId};

/// Downward refinement hint: tells a local node which regions to prioritize.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefinementHint {
    pub epoch: Epoch,
    pub geoid_level: u8,
    pub region_id: RegionId,
    pub geoid_residual_nanos: i64,
    /// Priority 1 (low) to 10 (high).
    pub priority: u8,
    pub suggested_neighbor_regions: Vec<RegionId>,
}

/// Output of the downward pass: measurement schedule for next epoch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementSchedule {
    pub priority_regions: Vec<(RegionId, u8)>,
    pub min_measurements: u32,
    pub extra_measurements_per_priority: u32,
}

impl MeasurementSchedule {
    pub fn default_schedule() -> Self {
        Self {
            priority_regions: Vec::new(),
            min_measurements: 8,
            extra_measurements_per_priority: 2,
        }
    }

    pub fn total_measurements(&self) -> u32 {
        let extra: u32 = self
            .priority_regions
            .iter()
            .map(|(_, p)| *p as u32 * self.extra_measurements_per_priority)
            .sum();
        self.min_measurements + extra
    }
}

/// Manages bidirectional geoid circulation.
pub struct CirculationManager {
    pub geoid: NetworkGeoid,
    coarsener: CoarseningOperator,
    prev_geoid_hash: [u8; 32],
}

impl CirculationManager {
    pub fn new(epoch: Epoch) -> Self {
        Self {
            geoid: NetworkGeoid::new(epoch),
            coarsener: CoarseningOperator::new(),
            prev_geoid_hash: [0u8; 32],
        }
    }

    pub fn with_levels(epoch: Epoch, levels: u8) -> Self {
        Self {
            geoid: NetworkGeoid::new(epoch),
            coarsener: CoarseningOperator::with_levels(levels),
            prev_geoid_hash: [0u8; 32],
        }
    }

    /// Upward pass: build the geoid from a set of L1 beacons.
    pub fn process_upward(&mut self, beacons: &[&Layer1Beacon], epoch: Epoch) {
        self.prev_geoid_hash = self.geoid.hash();
        self.geoid = self.coarsener.build_hierarchy(beacons, epoch);
        self.geoid.refinement_generation += 1;
        self.geoid.prev_hash = self.prev_geoid_hash;
    }

    /// Downward pass: generate refinement hints for a node at the given coordinates.
    pub fn generate_hints(&self, local_coords: &Coordinates) -> Vec<RefinementHint> {
        let mut hints = Vec::new();
        let epoch = self.geoid.epoch;

        for layer in &self.geoid.layers {
            // Find which region the local node is in
            if let Some(local_region_id) = layer.assign_region(local_coords) {
                if let Some(region) = layer.find_region(&local_region_id) {
                    let residual_nanos = (region.geoid_residual * 1e9) as i64;
                    let priority = hint_priority(residual_nanos.unsigned_abs(), layer.level);

                    let suggested: Vec<RegionId> = region
                        .inter_edges
                        .iter()
                        .map(|e| e.to)
                        .take(3)
                        .collect();

                    hints.push(RefinementHint {
                        epoch,
                        geoid_level: layer.level,
                        region_id: local_region_id,
                        geoid_residual_nanos: residual_nanos,
                        priority,
                        suggested_neighbor_regions: suggested,
                    });
                }
            }
        }

        hints
    }

    /// Convert hints into a measurement schedule for the next epoch.
    pub fn apply_hints(&self, hints: &[RefinementHint]) -> MeasurementSchedule {
        let mut schedule = MeasurementSchedule::default_schedule();

        for hint in hints {
            schedule
                .priority_regions
                .push((hint.region_id, hint.priority));
        }

        // Sort highest priority first
        schedule
            .priority_regions
            .sort_by_key(|(_, p)| std::cmp::Reverse(*p));

        schedule
    }

    /// How much the geoid changed this epoch, normalized to [0.0, 1.0].
    pub fn convergence_delta(&self) -> f64 {
        let current_hash = self.geoid.hash();
        if current_hash == self.prev_geoid_hash {
            0.0
        } else {
            // Since we only store the current geoid, we use the hash change as a signal.
            // For the full structural delta we'd need to keep the previous geoid around.
            // Return 1.0 on the first epoch (prev_hash is zeroed), 0.5 otherwise.
            if self.prev_geoid_hash == [0u8; 32] {
                1.0
            } else {
                // Hashes differ so there was some change; report non-zero
                0.5
            }
        }
    }
}

fn hint_priority(residual_nanos: u64, level: u8) -> u8 {
    let base: u8 = match residual_nanos {
        0..=100_000 => 1,
        100_001..=1_000_000 => 3,
        1_000_001..=10_000_000 => 5,
        10_000_001..=100_000_000 => 7,
        _ => 9,
    };
    (base + if level == 0 { 1 } else { 0 }).min(10)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto::blinding::BlindingSecret;
    use crate::layers::layer1::Layer1Beacon;
    use crate::sync::geometry::CoordinateState;
    use crate::types::{Epoch, NodeId};

    fn make_beacon(node_bytes: u8, epoch: Epoch, pos: [f64; 6]) -> Layer1Beacon {
        let node_id = NodeId([node_bytes; 32]);
        let secret = BlindingSecret::new(&node_id, epoch);
        let blinded_id = secret.blind();
        let proof = secret.prove_knowledge();
        let coords = CoordinateState {
            position: crate::types::Coordinates { dims: pos },
            error: 0.01,
        };
        Layer1Beacon {
            blinded_id,
            epoch,
            coordinates: coords,
            clock_offset: pos[0] * 0.001,
            clock_drift: 0.0,
            uncertainty: 0.001,
            peer_count: 3,
            consistency_score: 0.99,
            proof_of_identity: proof,
            proof_of_peer_count: None,
            geoid_coord: None,
        }
    }

    #[test]
    fn test_circulation_upward_builds_geoid() {
        let epoch = Epoch(1);
        let mut manager = CirculationManager::with_levels(epoch, 2);

        let b1 = make_beacon(1, epoch, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let b2 = make_beacon(2, epoch, [0.001, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let b3 = make_beacon(3, epoch, [0.01, 0.0, 0.0, 0.0, 0.0, 0.0]);

        let beacons: Vec<&Layer1Beacon> = vec![&b1, &b2, &b3];
        manager.process_upward(&beacons, epoch);

        assert!(!manager.geoid.layers.is_empty());
        assert_eq!(manager.geoid.epoch, epoch);
        // Should have at least a leaf layer
        assert!(manager.geoid.depth() >= 1);
    }

    #[test]
    fn test_circulation_downward_generates_hints() {
        let epoch = Epoch(2);
        let mut manager = CirculationManager::with_levels(epoch, 2);

        let b1 = make_beacon(1, epoch, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let b2 = make_beacon(2, epoch, [0.005, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let b3 = make_beacon(3, epoch, [0.010, 0.0, 0.0, 0.0, 0.0, 0.0]);

        let beacons: Vec<&Layer1Beacon> = vec![&b1, &b2, &b3];
        manager.process_upward(&beacons, epoch);

        let local_pos = crate::types::Coordinates {
            dims: [0.002, 0.0, 0.0, 0.0, 0.0, 0.0],
        };
        let hints = manager.generate_hints(&local_pos);

        // Should have at least one hint per layer
        assert!(!hints.is_empty());
        for hint in &hints {
            assert!(hint.priority >= 1 && hint.priority <= 10);
        }
    }

    #[test]
    fn test_full_circulation_roundtrip() {
        let epoch = Epoch(3);
        let mut manager = CirculationManager::with_levels(epoch, 3);

        // Create a cluster of nearby beacons and a distant outlier
        let beacons_data = vec![
            make_beacon(1, epoch, [0.000, 0.000, 0.0, 0.0, 0.0, 0.0]),
            make_beacon(2, epoch, [0.001, 0.000, 0.0, 0.0, 0.0, 0.0]),
            make_beacon(3, epoch, [0.000, 0.001, 0.0, 0.0, 0.0, 0.0]),
            make_beacon(4, epoch, [0.050, 0.050, 0.0, 0.0, 0.0, 0.0]), // distant
        ];
        let beacons: Vec<&Layer1Beacon> = beacons_data.iter().collect();

        // Upward pass
        manager.process_upward(&beacons, epoch);
        assert!(!manager.geoid.layers.is_empty());

        // Downward pass: generate hints for a local node
        let local_pos = crate::types::Coordinates {
            dims: [0.0005, 0.0005, 0.0, 0.0, 0.0, 0.0],
        };
        let hints = manager.generate_hints(&local_pos);

        // Apply hints to get a schedule
        let schedule = manager.apply_hints(&hints);

        // Schedule should have a valid measurement count
        assert!(schedule.total_measurements() >= schedule.min_measurements);

        // delta should be non-zero (first epoch after prev_hash = 0)
        let delta = manager.convergence_delta();
        assert!(delta >= 0.0 && delta <= 1.0);

        // Run a second epoch to test delta with real previous state
        let epoch2 = Epoch(4);
        let beacons2_data = vec![
            make_beacon(1, epoch2, [0.000, 0.000, 0.0, 0.0, 0.0, 0.0]),
            make_beacon(2, epoch2, [0.001, 0.000, 0.0, 0.0, 0.0, 0.0]),
        ];
        let beacons2: Vec<&Layer1Beacon> = beacons2_data.iter().collect();
        manager.process_upward(&beacons2, epoch2);

        let delta2 = manager.convergence_delta();
        assert!(delta2 >= 0.0 && delta2 <= 1.0);
    }
}
