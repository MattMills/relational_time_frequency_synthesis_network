use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::types::NodeId;

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
}
