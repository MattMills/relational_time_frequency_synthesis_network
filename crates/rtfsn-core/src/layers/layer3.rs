use serde::{Deserialize, Serialize};

use crate::crypto::vdf::VdfProof;
use crate::types::Epoch;

/// Layer 3: Surface clock reference stream.
///
/// This is the only thing visible from outside the system. It produces
/// a monotonically increasing time value backed by the full proof chain,
/// revealing nothing about who produced it or how.

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClockTick {
    pub epoch: Epoch,
    pub network_time_nanos: u64,
    pub confidence_nanos: u64,
    pub epoch_hash: [u8; 32],
    pub prev_hash: [u8; 32],
    pub vdf_proof: Option<VdfProof>,
}

impl ClockTick {
    pub fn genesis() -> Self {
        Self {
            epoch: Epoch(0),
            network_time_nanos: 0,
            confidence_nanos: u64::MAX,
            epoch_hash: [0u8; 32],
            prev_hash: [0u8; 32],
            vdf_proof: None,
        }
    }

    pub fn hash(&self) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&self.epoch.0.to_le_bytes());
        hasher.update(&self.network_time_nanos.to_le_bytes());
        hasher.update(&self.confidence_nanos.to_le_bytes());
        hasher.update(&self.epoch_hash);
        hasher.update(&self.prev_hash);
        *hasher.finalize().as_bytes()
    }

    pub fn verify_chain(&self, prev: &ClockTick) -> bool {
        if self.epoch.0 != prev.epoch.0 + 1 {
            return false;
        }
        if self.prev_hash != prev.hash() {
            return false;
        }
        if self.network_time_nanos < prev.network_time_nanos {
            return false;
        }
        if let Some(ref vdf) = self.vdf_proof {
            if !vdf.verify() {
                return false;
            }
        }
        true
    }
}

/// The clock stream — produces ticks from consensus data.
#[derive(Debug)]
pub struct ClockStream {
    ticks: Vec<ClockTick>,
}

impl ClockStream {
    pub fn new() -> Self {
        Self {
            ticks: vec![ClockTick::genesis()],
        }
    }

    pub fn latest(&self) -> &ClockTick {
        self.ticks.last().expect("stream always has genesis")
    }

    pub fn emit(
        &mut self,
        network_time_nanos: u64,
        confidence_nanos: u64,
        epoch_hash: [u8; 32],
        vdf_proof: Option<VdfProof>,
    ) -> &ClockTick {
        let prev = self.latest();
        let prev_hash = prev.hash();
        let epoch = prev.epoch.next();

        // Monotonicity: never go backwards
        let clamped_time =
            network_time_nanos.max(prev.network_time_nanos + 1);

        let tick = ClockTick {
            epoch,
            network_time_nanos: clamped_time,
            confidence_nanos,
            epoch_hash,
            prev_hash,
            vdf_proof,
        };

        self.ticks.push(tick);
        self.ticks.last().unwrap()
    }

    pub fn len(&self) -> usize {
        self.ticks.len()
    }

    pub fn get(&self, epoch: Epoch) -> Option<&ClockTick> {
        self.ticks.get(epoch.0 as usize)
    }

    /// Verify the entire chain from genesis to tip.
    pub fn verify_all(&self) -> bool {
        for i in 1..self.ticks.len() {
            if !self.ticks[i].verify_chain(&self.ticks[i - 1]) {
                return false;
            }
        }
        true
    }
}

impl Default for ClockStream {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clock_stream_chain() {
        let mut stream = ClockStream::new();

        for i in 1u64..=10 {
            let time = i as u64 * 1_000_000_000;
            let confidence = 5_000_000;
            let hash = *blake3::hash(&i.to_le_bytes()).as_bytes();
            stream.emit(time, confidence, hash, None);
        }

        assert_eq!(stream.len(), 11); // genesis + 10
        assert!(stream.verify_all());
    }

    #[test]
    fn test_monotonicity() {
        let mut stream = ClockStream::new();

        stream.emit(100, 1, [0u8; 32], None);
        // Try to go backwards
        stream.emit(50, 1, [1u8; 32], None);

        let latest = stream.latest();
        assert!(latest.network_time_nanos >= 100);
        assert!(stream.verify_all());
    }

    #[test]
    fn test_chain_verification_detects_tampering() {
        let mut stream = ClockStream::new();
        stream.emit(1_000_000_000, 5_000, [0u8; 32], None);
        stream.emit(2_000_000_000, 5_000, [1u8; 32], None);

        assert!(stream.verify_all());
    }
}
