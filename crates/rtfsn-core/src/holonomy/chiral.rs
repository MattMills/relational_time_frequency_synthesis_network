use serde::{Deserialize, Serialize};

/// ChiralFrame: integer-basis representation of a clock node's state.
///
/// Maps the clock problem onto a (p,q) system:
///   Site 0: clock offset in nanoseconds (integer — no float drift)
///   Site 1: drift rate in parts-per-billion (integer)
///   Site 2: measurement confidence score (integer quality metric)
///
/// Chirality encodes the identity commitment (Z_4), which gets
/// destroyed after two zero-screen projections through DHT layers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChiralFrame {
    pub offset_nanos: i64,
    pub drift_ppb: i64,
    pub confidence: u32,
    pub chirality: Chirality,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Chirality {
    C0 = 0,
    C1 = 1,
    C2 = 2,
    C3 = 3,
}

impl Chirality {
    pub fn from_identity_hash(hash: &[u8; 32]) -> Self {
        match hash[0] & 0x03 {
            0 => Chirality::C0,
            1 => Chirality::C1,
            2 => Chirality::C2,
            _ => Chirality::C3,
        }
    }

    /// Project through zero screen: Z_4 → Z_2
    pub fn project_once(self) -> u8 {
        (self as u8) & 0x01
    }

    /// Two projections: Z_4 → Z_2 → Z_1 (trivial)
    /// Identity information is fully destroyed.
    pub fn project_twice(self) -> u8 {
        let _ = self.project_once();
        0
    }
}

impl ChiralFrame {
    pub fn new(offset_nanos: i64, drift_ppb: i64, confidence: u32) -> Self {
        Self {
            offset_nanos,
            drift_ppb,
            confidence,
            chirality: Chirality::C0,
        }
    }

    pub fn with_chirality(mut self, chirality: Chirality) -> Self {
        self.chirality = chirality;
        self
    }

    pub fn offset_secs(&self) -> f64 {
        self.offset_nanos as f64 / 1_000_000_000.0
    }

    pub fn drift_rate(&self) -> f64 {
        self.drift_ppb as f64 / 1_000_000_000.0
    }

    /// Predict the offset at a future time (delta in nanoseconds).
    pub fn predict_offset(&self, delta_nanos: i64) -> i64 {
        // offset + drift_ppb * delta / 1e9
        // Using integer arithmetic to avoid float accumulation:
        self.offset_nanos + (self.drift_ppb * delta_nanos) / 1_000_000_000
    }

    /// Twist between two frames: the residual after accounting for
    /// the predicted offset difference.
    pub fn twist(&self, other: &ChiralFrame) -> TwistValue {
        let offset_diff = self.offset_nanos - other.offset_nanos;
        let drift_diff = self.drift_ppb - other.drift_ppb;
        TwistValue {
            offset_diff_nanos: offset_diff,
            drift_diff_ppb: drift_diff,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TwistValue {
    pub offset_diff_nanos: i64,
    pub drift_diff_ppb: i64,
}

impl TwistValue {
    pub fn magnitude_nanos(&self) -> u64 {
        self.offset_diff_nanos.unsigned_abs()
    }

    pub fn is_consistent(&self, measured_rtt_nanos: u64, tolerance_nanos: u64) -> bool {
        // The offset difference should be explainable by the RTT
        // (asymmetric latency can account for up to RTT/2 of offset error)
        let max_offset_error = measured_rtt_nanos / 2 + tolerance_nanos;
        self.offset_diff_nanos.unsigned_abs() <= max_offset_error
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chiral_frame_predict() {
        let frame = ChiralFrame::new(1_000_000, 100, 10); // 1ms offset, 100ppb drift
        let predicted = frame.predict_offset(1_000_000_000); // 1 second later
        assert_eq!(predicted, 1_000_100); // offset + 100ns drift
    }

    #[test]
    fn test_twist_consistency() {
        let a = ChiralFrame::new(1_000_000, 0, 10);
        let b = ChiralFrame::new(1_005_000, 0, 10);
        let twist = a.twist(&b);

        // 5μs offset difference, RTT of 10ms → consistent
        assert!(twist.is_consistent(10_000_000, 1_000));
        // RTT of 1μs → inconsistent (offset diff exceeds RTT/2)
        assert!(!twist.is_consistent(1_000, 1_000));
    }

    #[test]
    fn test_chirality_projection() {
        assert_eq!(Chirality::C3.project_once(), 1);
        assert_eq!(Chirality::C2.project_once(), 0);
        // Two projections always yield trivial
        for c in [Chirality::C0, Chirality::C1, Chirality::C2, Chirality::C3] {
            assert_eq!(c.project_twice(), 0);
        }
    }
}
