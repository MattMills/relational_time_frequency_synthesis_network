use serde::{Deserialize, Serialize};

pub const COORDINATE_DIMENSIONS: usize = 6;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Epoch(pub u64);

impl Epoch {
    pub fn next(self) -> Self {
        Self(self.0 + 1)
    }

    pub fn prev(self) -> Option<Self> {
        self.0.checked_sub(1).map(Self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeId(pub [u8; 32]);

impl NodeId {
    pub fn from_public_key(pk: &ed25519_dalek::VerifyingKey) -> Self {
        let hash = blake3::hash(pk.as_bytes());
        Self(*hash.as_bytes())
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Timestamp {
    pub nanos: u64,
}

impl Timestamp {
    pub fn now() -> Self {
        #[cfg(not(target_arch = "wasm32"))]
        {
            use std::time::{SystemTime, UNIX_EPOCH};
            let d = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
            Self {
                nanos: d.as_nanos() as u64,
            }
        }
        #[cfg(target_arch = "wasm32")]
        {
            Self { nanos: 0 }
        }
    }

    pub fn from_nanos(nanos: u64) -> Self {
        Self { nanos }
    }

    pub fn as_secs_f64(&self) -> f64 {
        self.nanos as f64 / 1_000_000_000.0
    }

    pub fn diff_secs(&self, other: &Self) -> f64 {
        (self.nanos as f64 - other.nanos as f64) / 1_000_000_000.0
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Duration {
    pub nanos: u64,
}

impl Duration {
    pub fn from_nanos(nanos: u64) -> Self {
        Self { nanos }
    }

    pub fn as_secs_f64(&self) -> f64 {
        self.nanos as f64 / 1_000_000_000.0
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Coordinates {
    pub dims: [f64; COORDINATE_DIMENSIONS],
}

impl Coordinates {
    pub fn origin() -> Self {
        Self {
            dims: [0.0; COORDINATE_DIMENSIONS],
        }
    }

    pub fn distance(&self, other: &Self) -> f64 {
        self.dims
            .iter()
            .zip(other.dims.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    pub fn add_scaled(&self, direction: &Coordinates, scale: f64) -> Self {
        let mut dims = [0.0; COORDINATE_DIMENSIONS];
        for i in 0..COORDINATE_DIMENSIONS {
            dims[i] = self.dims[i] + direction.dims[i] * scale;
        }
        Self { dims }
    }

    pub fn subtract(&self, other: &Coordinates) -> Self {
        let mut dims = [0.0; COORDINATE_DIMENSIONS];
        for i in 0..COORDINATE_DIMENSIONS {
            dims[i] = self.dims[i] - other.dims[i];
        }
        Self { dims }
    }

    pub fn normalize(&self) -> Self {
        let mag = self.magnitude();
        if mag < 1e-12 {
            return Self::origin();
        }
        let mut dims = [0.0; COORDINATE_DIMENSIONS];
        for i in 0..COORDINATE_DIMENSIONS {
            dims[i] = self.dims[i] / mag;
        }
        Self { dims }
    }

    pub fn magnitude(&self) -> f64 {
        self.dims.iter().map(|d| d.powi(2)).sum::<f64>().sqrt()
    }
}

/// Statistical profile of latency distribution between two nodes over multiple epochs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationalLatencyProfile {
    pub mean_nanos: i64,
    pub variance_nanos: i64,
    pub p10_nanos: i64,
    pub p50_nanos: i64,
    pub p90_nanos: i64,
    /// Rate of change of mean latency (nanoseconds per epoch, positive = increasing)
    pub trend_nanos_per_epoch: i64,
    pub sample_count: u32,
    pub epoch_first: u64,
    pub epoch_last: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MerkleDigest(pub [u8; 32]);

impl MerkleDigest {
    pub fn from_items(items: &[&[u8]]) -> Self {
        let mut hasher = blake3::Hasher::new();
        for item in items {
            let item_hash = blake3::hash(item);
            hasher.update(item_hash.as_bytes());
        }
        Self(*hasher.finalize().as_bytes())
    }

    pub fn combine(left: &Self, right: &Self) -> Self {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&left.0);
        hasher.update(&right.0);
        Self(*hasher.finalize().as_bytes())
    }
}
