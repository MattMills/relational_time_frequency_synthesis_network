use serde::{Deserialize, Serialize};

/// SAMR (Structured Adaptive Multi-Resolution) prime-channel scheduler.
///
/// Each prime channel captures one dimension of the clock state at
/// a different resolution. Channels are independent; cross-channel
/// coupling occurs only at CRT (Chinese Remainder Theorem) boundaries.
///
/// Channel p=2: Binary — is the node reachable? (1 round trip)
/// Channel p=3: Ternary — network tier (fast/medium/slow)
/// Channel p=5: Quinary — clock offset precision (5 levels)
/// Channel p=7: Septenary — drift rate categories (7 levels)
///
/// Total resolution: 2 × 3 × 5 × 7 = 210 distinguishable states
/// after just 4 channel refinements.

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrimeChannel {
    pub prime: u32,
    pub value: u32,
    pub defect: u64,
}

impl PrimeChannel {
    pub fn new(prime: u32) -> Self {
        Self {
            prime,
            value: 0,
            defect: u64::MAX,
        }
    }

    pub fn set(&mut self, value: u32, defect: u64) {
        assert!(value < self.prime, "value must be < prime");
        self.value = value;
        self.defect = defect;
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SAMRState {
    pub channels: Vec<PrimeChannel>,
}

impl SAMRState {
    pub fn clock_default() -> Self {
        Self {
            channels: vec![
                PrimeChannel::new(2), // reachability
                PrimeChannel::new(3), // network tier
                PrimeChannel::new(5), // offset precision
                PrimeChannel::new(7), // drift category
            ],
        }
    }

    /// Total resolution: product of all primes
    pub fn total_resolution(&self) -> u64 {
        self.channels.iter().map(|c| c.prime as u64).product()
    }

    /// CRT decode: combine independent channel values into a single
    /// integer in [0, total_resolution).
    pub fn crt_decode(&self) -> u64 {
        let n = self.total_resolution();
        let mut result: u64 = 0;

        for ch in &self.channels {
            let ni = n / ch.prime as u64;
            let yi = mod_inverse(ni, ch.prime as u64).unwrap_or(0);
            result += ch.value as u64 * ni * yi;
        }

        result % n
    }

    /// Which channel has the highest defect? Refine that one next.
    pub fn next_channel_to_refine(&self) -> Option<usize> {
        self.channels
            .iter()
            .enumerate()
            .max_by_key(|(_, ch)| ch.defect)
            .map(|(i, _)| i)
    }

    /// Classify a node's reachability (channel 0, p=2)
    pub fn classify_reachability(&mut self, reachable: bool) {
        self.channels[0].set(if reachable { 1 } else { 0 }, 0);
    }

    /// Classify network tier (channel 1, p=3)
    pub fn classify_tier(&mut self, rtt_nanos: u64) {
        let tier = if rtt_nanos < 10_000_000 {
            0 // fast: < 10ms
        } else if rtt_nanos < 100_000_000 {
            1 // medium: < 100ms
        } else {
            2 // slow: >= 100ms
        };
        self.channels[1].set(tier, 0);
    }

    /// Classify offset precision (channel 2, p=5)
    pub fn classify_offset(&mut self, offset_nanos: i64, uncertainty_nanos: u64) {
        let level = if uncertainty_nanos < 1_000_000 {
            0 // sub-ms
        } else if uncertainty_nanos < 5_000_000 {
            1 // 1-5ms
        } else if uncertainty_nanos < 20_000_000 {
            2 // 5-20ms
        } else if uncertainty_nanos < 100_000_000 {
            3 // 20-100ms
        } else {
            4 // > 100ms
        };
        let _ = offset_nanos; // used for defect calculation in full impl
        self.channels[2].set(level, uncertainty_nanos);
    }

    /// Classify drift category (channel 3, p=7)
    pub fn classify_drift(&mut self, drift_ppb: i64) {
        let abs_drift = drift_ppb.unsigned_abs();
        let category = if abs_drift < 10 {
            3 // centered: < 10 ppb
        } else if drift_ppb > 0 {
            if abs_drift < 100 {
                4 // slight fast
            } else if abs_drift < 1000 {
                5 // moderate fast
            } else {
                6 // severe fast
            }
        } else if abs_drift < 100 {
            2 // slight slow
        } else if abs_drift < 1000 {
            1 // moderate slow
        } else {
            0 // severe slow
        };
        self.channels[3].set(category, abs_drift);
    }
}

fn mod_inverse(a: u64, m: u64) -> Option<u64> {
    if m == 1 {
        return Some(0);
    }
    let (mut old_r, mut r) = (a as i128, m as i128);
    let (mut old_s, mut s) = (1i128, 0i128);

    while r != 0 {
        let q = old_r / r;
        let temp_r = r;
        r = old_r - q * r;
        old_r = temp_r;
        let temp_s = s;
        s = old_s - q * s;
        old_s = temp_s;
    }

    if old_r != 1 {
        return None;
    }

    Some(((old_s % m as i128 + m as i128) % m as i128) as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_samr_total_resolution() {
        let state = SAMRState::clock_default();
        assert_eq!(state.total_resolution(), 210);
    }

    #[test]
    fn test_crt_roundtrip() {
        let mut state = SAMRState::clock_default();
        state.channels[0].set(1, 0); // reachable
        state.channels[1].set(0, 0); // fast
        state.channels[2].set(2, 0); // 5-20ms precision
        state.channels[3].set(3, 0); // centered drift

        let encoded = state.crt_decode();
        assert!(encoded < 210);

        // Decode back by modular arithmetic
        assert_eq!(encoded % 2, 1);
        assert_eq!(encoded % 3, 0);
        assert_eq!(encoded % 5, 2);
        assert_eq!(encoded % 7, 3);
    }

    #[test]
    fn test_adaptive_scheduling() {
        let mut state = SAMRState::clock_default();
        state.channels[0].set(1, 0);
        state.channels[1].set(1, 50_000_000);
        state.channels[2].set(2, 10_000_000);
        state.channels[3].set(3, 200);

        let next = state.next_channel_to_refine().unwrap();
        assert_eq!(next, 1, "should refine tier channel (highest defect)");
    }
}
