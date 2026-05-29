use serde::{Deserialize, Serialize};
use std::collections::HashSet;

use crate::types::{Epoch, NodeId};

/// Epoch phase within the fractal state pump.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EpochPhase {
    /// Phase 1: Private pairwise measurements
    Measure,
    /// Phase 2: Local solve using measurements + prior beacons
    Solve,
    /// Phase 3: Publish beacon, wait for quorum
    Publish,
}

impl EpochPhase {
    pub fn next(self) -> Option<Self> {
        match self {
            EpochPhase::Measure => Some(EpochPhase::Solve),
            EpochPhase::Solve => Some(EpochPhase::Publish),
            EpochPhase::Publish => None, // advance to next epoch
        }
    }
}

/// Manages the epoch lifecycle for this node.
#[derive(Debug)]
pub struct EpochManager {
    pub current_epoch: Epoch,
    pub current_phase: EpochPhase,
    pub quorum_threshold: usize,
    received_beacons: HashSet<NodeId>,
    phase_start_time: u64,
    pub phase_timeout_nanos: u64,
}

impl EpochManager {
    pub fn new(quorum_threshold: usize, phase_timeout_nanos: u64) -> Self {
        Self {
            current_epoch: Epoch(0),
            current_phase: EpochPhase::Measure,
            quorum_threshold,
            received_beacons: HashSet::new(),
            phase_start_time: 0,
            phase_timeout_nanos,
        }
    }

    /// Try to advance the phase. Returns true if we advanced.
    pub fn try_advance(&mut self, current_time_nanos: u64) -> bool {
        let timed_out =
            current_time_nanos - self.phase_start_time > self.phase_timeout_nanos;

        match self.current_phase {
            EpochPhase::Measure => {
                // Advance to Solve when we have enough measurements or timeout
                if timed_out {
                    self.current_phase = EpochPhase::Solve;
                    self.phase_start_time = current_time_nanos;
                    return true;
                }
            }
            EpochPhase::Solve => {
                // Advance to Publish when solve completes (always immediate)
                self.current_phase = EpochPhase::Publish;
                self.phase_start_time = current_time_nanos;
                return true;
            }
            EpochPhase::Publish => {
                // Advance to next epoch when quorum reached or timeout
                if self.received_beacons.len() >= self.quorum_threshold
                    || timed_out
                {
                    self.current_epoch = self.current_epoch.next();
                    self.current_phase = EpochPhase::Measure;
                    self.received_beacons.clear();
                    self.phase_start_time = current_time_nanos;
                    return true;
                }
            }
        }

        false
    }

    /// Record that we received a beacon from a peer for the current epoch.
    pub fn record_beacon(&mut self, peer: NodeId) {
        self.received_beacons.insert(peer);
    }

    pub fn beacon_count(&self) -> usize {
        self.received_beacons.len()
    }

    pub fn has_quorum(&self) -> bool {
        self.received_beacons.len() >= self.quorum_threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_epoch_lifecycle() {
        let mut mgr = EpochManager::new(3, 10_000_000_000); // 10s timeout

        assert_eq!(mgr.current_epoch, Epoch(0));
        assert_eq!(mgr.current_phase, EpochPhase::Measure);

        // Timeout measure phase
        assert!(mgr.try_advance(20_000_000_000));
        assert_eq!(mgr.current_phase, EpochPhase::Solve);

        // Solve immediately advances to publish
        assert!(mgr.try_advance(20_000_000_001));
        assert_eq!(mgr.current_phase, EpochPhase::Publish);

        // Need quorum to advance
        assert!(!mgr.try_advance(20_000_000_002));

        mgr.record_beacon(NodeId([1u8; 32]));
        mgr.record_beacon(NodeId([2u8; 32]));
        mgr.record_beacon(NodeId([3u8; 32]));

        assert!(mgr.try_advance(20_000_000_003));
        assert_eq!(mgr.current_epoch, Epoch(1));
        assert_eq!(mgr.current_phase, EpochPhase::Measure);
    }
}
