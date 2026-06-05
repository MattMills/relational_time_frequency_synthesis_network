use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::types::{NodeId, RelationalLatencyProfile};

/// TwistIndex: integer-basis encoding of the measurement between two nodes.
/// Stored at the connecting edge, not at either node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TwistIndex {
    pub rtt_nanos: u64,
    pub offset_nanos: i64,
    pub asymmetry_nanos: i64,
    pub quality: u32,
    pub epoch_measured: u64,
}

impl TwistIndex {
    pub fn from_exchange(
        t1_nanos: u64,
        t2_nanos: u64,
        t3_nanos: u64,
        t4_nanos: u64,
    ) -> Self {
        let rtt = (t4_nanos.wrapping_sub(t1_nanos))
            .wrapping_sub(t3_nanos.wrapping_sub(t2_nanos));
        let offset = ((t2_nanos as i128 - t1_nanos as i128)
            + (t3_nanos as i128 - t4_nanos as i128))
            / 2;
        let forward = t2_nanos as i128 - t1_nanos as i128;
        let backward = t4_nanos as i128 - t3_nanos as i128;
        let asymmetry = forward - backward;

        Self {
            rtt_nanos: rtt,
            offset_nanos: offset as i64,
            asymmetry_nanos: asymmetry as i64,
            quality: 100,
            epoch_measured: 0,
        }
    }

    pub fn rtt_secs(&self) -> f64 {
        self.rtt_nanos as f64 / 1_000_000_000.0
    }

    pub fn offset_secs(&self) -> f64 {
        self.offset_nanos as f64 / 1_000_000_000.0
    }
}

/// TwistLUT: the lookup table of all pairwise measurements.
/// Stored per-edge; each node only sees its own edges.
#[derive(Debug)]
pub struct TwistLUT {
    entries: HashMap<(NodeId, NodeId), Vec<TwistIndex>>,
}

impl TwistLUT {
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    fn edge_key(a: NodeId, b: NodeId) -> (NodeId, NodeId) {
        if a.0 < b.0 {
            (a, b)
        } else {
            (b, a)
        }
    }

    pub fn insert(&mut self, a: NodeId, b: NodeId, twist: TwistIndex) {
        let key = Self::edge_key(a, b);
        self.entries.entry(key).or_default().push(twist);
    }

    pub fn get(&self, a: NodeId, b: NodeId) -> Option<&[TwistIndex]> {
        let key = Self::edge_key(a, b);
        self.entries.get(&key).map(|v| v.as_slice())
    }

    pub fn latest(&self, a: NodeId, b: NodeId) -> Option<&TwistIndex> {
        self.get(a, b)?.last()
    }

    pub fn neighbors(&self, node: NodeId) -> Vec<NodeId> {
        self.entries
            .keys()
            .filter_map(|(a, b)| {
                if *a == node {
                    Some(*b)
                } else if *b == node {
                    Some(*a)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Compute holonomy around a loop: sum of offsets should be ~zero
    /// for a consistent geometry. Non-zero holonomy = defect.
    pub fn loop_holonomy(&self, path: &[NodeId]) -> Option<i64> {
        if path.len() < 3 {
            return None;
        }

        let mut total: i64 = 0;
        for i in 0..path.len() {
            let a = path[i];
            let b = path[(i + 1) % path.len()];
            let twist = self.latest(a, b)?;
            // Direction matters: if we stored (a,b) but traverse (b,a), negate
            if Self::edge_key(a, b).0 == a {
                total += twist.offset_nanos;
            } else {
                total -= twist.offset_nanos;
            }
        }

        Some(total)
    }

    pub fn edge_count(&self) -> usize {
        self.entries.len()
    }

    /// Compute a relational latency profile from the measurement history for an edge.
    pub fn profile(&self, a: NodeId, b: NodeId) -> Option<RelationalLatencyProfile> {
        let twists = self.get(a, b)?;
        if twists.is_empty() {
            return None;
        }

        // Compute mean
        let mean = twists.iter().map(|t| t.rtt_nanos as i64).sum::<i64>()
            / twists.len() as i64;

        // Compute variance
        let variance = twists
            .iter()
            .map(|t| {
                let d = t.rtt_nanos as i64 - mean;
                d * d
            })
            .sum::<i64>()
            / twists.len() as i64;

        // Compute percentiles (sort RTTs)
        let mut rtts: Vec<i64> =
            twists.iter().map(|t| t.rtt_nanos as i64).collect();
        rtts.sort_unstable();
        let p10 = rtts[rtts.len() / 10];
        let p50 = rtts[rtts.len() / 2];
        let p90 = rtts[rtts.len() * 9 / 10];

        // Compute trend using linear regression on epoch_measured vs rtt
        let trend = if twists.len() >= 2 {
            let n = twists.len() as i64;
            let sum_x: i64 =
                twists.iter().map(|t| t.epoch_measured as i64).sum();
            let sum_y: i64 =
                twists.iter().map(|t| t.rtt_nanos as i64).sum();
            let sum_xy: i64 = twists
                .iter()
                .map(|t| t.epoch_measured as i64 * t.rtt_nanos as i64)
                .sum();
            let sum_xx: i64 = twists
                .iter()
                .map(|t| (t.epoch_measured as i64).pow(2))
                .sum();
            let denom = n * sum_xx - sum_x * sum_x;
            if denom != 0 {
                (n * sum_xy - sum_x * sum_y) / denom
            } else {
                0
            }
        } else {
            0
        };

        let epoch_first =
            twists.first().map(|t| t.epoch_measured).unwrap_or(0);
        let epoch_last =
            twists.last().map(|t| t.epoch_measured).unwrap_or(0);

        Some(RelationalLatencyProfile {
            mean_nanos: mean,
            variance_nanos: variance,
            p10_nanos: p10,
            p50_nanos: p50,
            p90_nanos: p90,
            trend_nanos_per_epoch: trend,
            sample_count: twists.len() as u32,
            epoch_first,
            epoch_last,
        })
    }
}

impl Default for TwistLUT {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_twist_lut_basic() {
        let mut lut = TwistLUT::new();
        let a = NodeId([1u8; 32]);
        let b = NodeId([2u8; 32]);

        let twist = TwistIndex {
            rtt_nanos: 50_000_000,
            offset_nanos: 1_000_000,
            asymmetry_nanos: 100_000,
            quality: 100,
            epoch_measured: 1,
        };

        lut.insert(a, b, twist);
        assert!(lut.latest(a, b).is_some());
        assert!(lut.latest(b, a).is_some()); // symmetric lookup
    }

    #[test]
    fn test_loop_holonomy_consistent() {
        let mut lut = TwistLUT::new();
        let a = NodeId([1u8; 32]);
        let b = NodeId([2u8; 32]);
        let c = NodeId([3u8; 32]);

        // Consistent triangle: a→b = +1ms, b→c = +2ms, c→a = -3ms
        lut.insert(a, b, TwistIndex {
            rtt_nanos: 10_000_000,
            offset_nanos: 1_000_000,
            asymmetry_nanos: 0,
            quality: 100,
            epoch_measured: 1,
        });
        lut.insert(b, c, TwistIndex {
            rtt_nanos: 10_000_000,
            offset_nanos: 2_000_000,
            asymmetry_nanos: 0,
            quality: 100,
            epoch_measured: 1,
        });
        lut.insert(a, c, TwistIndex {
            rtt_nanos: 10_000_000,
            offset_nanos: 3_000_000,
            asymmetry_nanos: 0,
            quality: 100,
            epoch_measured: 1,
        });

        let holonomy = lut.loop_holonomy(&[a, b, c]).unwrap();
        // Consistent: a→b(+1) + b→c(+2) + c→a(-3) = 0
        assert_eq!(holonomy, 0, "holonomy = {holonomy}");
    }

    #[test]
    fn test_relational_latency_profile() {
        let mut lut = TwistLUT::new();
        let a = NodeId([1u8; 32]);
        let b = NodeId([2u8; 32]);

        // Add multiple measurements with increasing RTT (trend > 0)
        let rtts: &[u64] = &[
            10_000_000, 12_000_000, 14_000_000, 16_000_000, 18_000_000,
            20_000_000, 22_000_000, 24_000_000, 26_000_000, 28_000_000,
        ];
        for (i, &rtt) in rtts.iter().enumerate() {
            lut.insert(
                a,
                b,
                TwistIndex {
                    rtt_nanos: rtt,
                    offset_nanos: 0,
                    asymmetry_nanos: 0,
                    quality: 100,
                    epoch_measured: (i + 1) as u64,
                },
            );
        }

        let profile = lut.profile(a, b).expect("profile should exist");
        assert_eq!(profile.sample_count, 10);
        // Mean of 10..28ms in steps of 2ms = 19ms
        assert_eq!(profile.mean_nanos, 19_000_000);
        // Percentiles: p50 should be in the middle
        assert!(profile.p50_nanos >= 14_000_000 && profile.p50_nanos <= 24_000_000);
        // p10 < p50 < p90
        assert!(profile.p10_nanos <= profile.p50_nanos);
        assert!(profile.p50_nanos <= profile.p90_nanos);
        // Positive trend (RTT increasing)
        assert!(profile.trend_nanos_per_epoch > 0);
        assert_eq!(profile.epoch_first, 1);
        assert_eq!(profile.epoch_last, 10);
    }
}
