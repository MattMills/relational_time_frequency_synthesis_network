
use crate::types::NodeId;

const K_BUCKET_SIZE: usize = 20;
const KEY_BITS: usize = 256;

fn xor_distance(a: &[u8; 32], b: &[u8; 32]) -> [u8; 32] {
    let mut result = [0u8; 32];
    for i in 0..32 {
        result[i] = a[i] ^ b[i];
    }
    result
}

fn leading_zeros(distance: &[u8; 32]) -> usize {
    let mut zeros = 0;
    for byte in distance {
        if *byte == 0 {
            zeros += 8;
        } else {
            zeros += byte.leading_zeros() as usize;
            break;
        }
    }
    zeros
}

fn bucket_index(local: &[u8; 32], remote: &[u8; 32]) -> usize {
    let dist = xor_distance(local, remote);
    let lz = leading_zeros(&dist);
    if lz >= KEY_BITS {
        KEY_BITS - 1
    } else {
        KEY_BITS - 1 - lz
    }
}

/// Kademlia-style routing table.
/// Each layer gets its own routing table with its own key space.
#[derive(Debug)]
pub struct RoutingTable {
    local_id: [u8; 32],
    buckets: Vec<Vec<NodeId>>,
}

impl RoutingTable {
    pub fn new(local_id: [u8; 32]) -> Self {
        Self {
            local_id,
            buckets: (0..KEY_BITS).map(|_| Vec::new()).collect(),
        }
    }

    pub fn insert(&mut self, node: NodeId) -> bool {
        if node.0 == self.local_id {
            return false;
        }

        let idx = bucket_index(&self.local_id, &node.0);
        let bucket = &mut self.buckets[idx];

        if bucket.contains(&node) {
            // Move to end (most recently seen)
            bucket.retain(|n| *n != node);
            bucket.push(node);
            return true;
        }

        if bucket.len() < K_BUCKET_SIZE {
            bucket.push(node);
            true
        } else {
            false // bucket full
        }
    }

    pub fn find_closest(&self, target: &[u8; 32], count: usize) -> Vec<NodeId> {
        let mut all: Vec<(NodeId, [u8; 32])> = self
            .buckets
            .iter()
            .flatten()
            .map(|n| (*n, xor_distance(&n.0, target)))
            .collect();

        all.sort_by(|a, b| a.1.cmp(&b.1));
        all.into_iter().take(count).map(|(n, _)| n).collect()
    }

    pub fn remove(&mut self, node: &NodeId) {
        for bucket in &mut self.buckets {
            bucket.retain(|n| n != node);
        }
    }

    pub fn node_count(&self) -> usize {
        self.buckets.iter().map(|b| b.len()).sum()
    }

    pub fn all_nodes(&self) -> Vec<NodeId> {
        self.buckets.iter().flatten().copied().collect()
    }
}

/// Multi-layer routing: each DHT layer has an independent routing table.
#[derive(Debug)]
pub struct LayeredRouting {
    pub layer0: RoutingTable,
    pub layer1: RoutingTable,
    pub layer2: RoutingTable,
    pub layer3: RoutingTable,
}

impl LayeredRouting {
    pub fn new(local_id: [u8; 32]) -> Self {
        Self {
            layer0: RoutingTable::new(local_id),
            layer1: RoutingTable::new(local_id),
            layer2: RoutingTable::new(local_id),
            layer3: RoutingTable::new(local_id),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_routing_table() {
        let local = [0u8; 32];
        let mut rt = RoutingTable::new(local);

        for i in 1..=30u8 {
            let mut id = [0u8; 32];
            id[0] = i;
            assert!(rt.insert(NodeId(id)));
        }

        assert_eq!(rt.node_count(), 30);

        let target = {
            let mut t = [0u8; 32];
            t[0] = 5;
            t
        };
        let closest = rt.find_closest(&target, 5);
        assert_eq!(closest.len(), 5);
        assert_eq!(closest[0].0[0], 5);
    }

    #[test]
    fn test_xor_distance_symmetry() {
        let a = [1u8; 32];
        let b = [2u8; 32];
        assert_eq!(xor_distance(&a, &b), xor_distance(&b, &a));
    }
}
