use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// Generic key-value storage for a single DHT layer.
/// Each layer uses this with different value types.
#[derive(Debug)]
pub struct LayerStorage {
    store: HashMap<[u8; 32], StoredValue>,
    max_entries: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredValue {
    pub data: Vec<u8>,
    pub timestamp: u64,
    pub ttl_epochs: u32,
    pub origin_epoch: u64,
}

impl LayerStorage {
    pub fn new(max_entries: usize) -> Self {
        Self {
            store: HashMap::new(),
            max_entries,
        }
    }

    pub fn put(&mut self, key: [u8; 32], value: StoredValue) -> bool {
        if self.store.len() >= self.max_entries
            && !self.store.contains_key(&key)
        {
            self.evict_oldest();
        }
        self.store.insert(key, value);
        true
    }

    pub fn get(&self, key: &[u8; 32]) -> Option<&StoredValue> {
        self.store.get(key)
    }

    pub fn remove(&mut self, key: &[u8; 32]) -> bool {
        self.store.remove(key).is_some()
    }

    pub fn len(&self) -> usize {
        self.store.len()
    }

    pub fn is_empty(&self) -> bool {
        self.store.is_empty()
    }

    pub fn expire(&mut self, current_epoch: u64) {
        self.store.retain(|_, v| {
            v.origin_epoch + v.ttl_epochs as u64 > current_epoch
        });
    }

    fn evict_oldest(&mut self) {
        if let Some(oldest_key) = self
            .store
            .iter()
            .min_by_key(|(_, v)| v.timestamp)
            .map(|(k, _)| *k)
        {
            self.store.remove(&oldest_key);
        }
    }

    pub fn keys(&self) -> impl Iterator<Item = &[u8; 32]> {
        self.store.keys()
    }
}

/// The complete 4-layer storage system.
#[derive(Debug)]
pub struct FourLayerStorage {
    pub layer0: LayerStorage,
    pub layer1: LayerStorage,
    pub layer2: LayerStorage,
    pub layer3: LayerStorage,
}

impl FourLayerStorage {
    pub fn new() -> Self {
        Self {
            layer0: LayerStorage::new(10_000),
            layer1: LayerStorage::new(50_000),
            layer2: LayerStorage::new(10_000),
            layer3: LayerStorage::new(1_000),
        }
    }

    pub fn expire_all(&mut self, current_epoch: u64) {
        self.layer0.expire(current_epoch);
        self.layer1.expire(current_epoch);
        self.layer2.expire(current_epoch);
        self.layer3.expire(current_epoch);
    }
}

impl Default for FourLayerStorage {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_storage_basic() {
        let mut storage = LayerStorage::new(100);

        let key = [1u8; 32];
        let value = StoredValue {
            data: vec![42],
            timestamp: 100,
            ttl_epochs: 10,
            origin_epoch: 1,
        };

        assert!(storage.put(key, value));
        assert!(storage.get(&key).is_some());
        assert_eq!(storage.len(), 1);
    }

    #[test]
    fn test_expiration() {
        let mut storage = LayerStorage::new(100);

        for i in 0..5u8 {
            let mut key = [0u8; 32];
            key[0] = i;
            storage.put(key, StoredValue {
                data: vec![i],
                timestamp: i as u64 * 100,
                ttl_epochs: 3,
                origin_epoch: i as u64,
            });
        }

        assert_eq!(storage.len(), 5);
        storage.expire(5);
        // Only entries with origin_epoch + ttl > 5 survive
        // origin 3 + ttl 3 = 6 > 5 ✓, origin 4 + ttl 3 = 7 > 5 ✓
        assert_eq!(storage.len(), 2);
    }
}
