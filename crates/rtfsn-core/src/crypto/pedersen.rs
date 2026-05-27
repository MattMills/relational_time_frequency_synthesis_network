use curve25519_dalek::constants::RISTRETTO_BASEPOINT_POINT;
use curve25519_dalek::ristretto::{CompressedRistretto, RistrettoPoint};
use curve25519_dalek::scalar::Scalar;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha512};

fn h_generator() -> RistrettoPoint {
    let hash = Sha512::digest(b"rtfsn_pedersen_h_generator_v1");
    RistrettoPoint::from_uniform_bytes(
        hash.as_slice().try_into().expect("SHA-512 produces 64 bytes"),
    )
}

#[derive(Debug, Clone)]
pub struct PedersenCommitment {
    pub point: RistrettoPoint,
    pub compressed: CompressedRistretto,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableCommitment {
    pub bytes: [u8; 32],
}

impl From<&PedersenCommitment> for SerializableCommitment {
    fn from(c: &PedersenCommitment) -> Self {
        Self {
            bytes: c.compressed.to_bytes(),
        }
    }
}

impl PedersenCommitment {
    /// C = g^value * h^blinding
    pub fn commit(value: &Scalar, blinding: &Scalar) -> Self {
        let g = RISTRETTO_BASEPOINT_POINT;
        let h = h_generator();
        let point = g * value + h * blinding;
        let compressed = point.compress();
        Self { point, compressed }
    }

    pub fn commit_f64(value: f64, blinding: &Scalar) -> Self {
        let quantized = (value * 1_000_000_000.0) as i64;
        let scalar = if quantized >= 0 {
            Scalar::from(quantized as u64)
        } else {
            -Scalar::from((-quantized) as u64)
        };
        Self::commit(&scalar, blinding)
    }

    pub fn add(&self, other: &PedersenCommitment) -> Self {
        let point = self.point + other.point;
        let compressed = point.compress();
        Self { point, compressed }
    }

    pub fn verify(&self, value: &Scalar, blinding: &Scalar) -> bool {
        let expected = Self::commit(value, blinding);
        self.point == expected.point
    }
}

/// Proof that a committed value lies within [0, 2^n)
/// Simplified range proof for prototyping — production would use Bulletproofs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RangeProof {
    pub commitment: SerializableCommitment,
    pub challenge: [u8; 32],
    pub response: [u8; 32],
    pub bit_count: u32,
}

impl RangeProof {
    pub fn create(value: u64, blinding: &Scalar, bit_count: u32) -> Option<Self> {
        if value >= (1u64 << bit_count.min(63)) {
            return None;
        }

        let value_scalar = Scalar::from(value);
        let commitment = PedersenCommitment::commit(&value_scalar, blinding);

        let mut hasher = blake3::Hasher::new();
        hasher.update(&commitment.compressed.to_bytes());
        hasher.update(&value.to_le_bytes());
        let challenge_hash = hasher.finalize();

        let challenge_scalar =
            Scalar::from_bytes_mod_order(*challenge_hash.as_bytes());
        let response = blinding - challenge_scalar * Scalar::from(value);

        Some(Self {
            commitment: SerializableCommitment::from(&commitment),
            challenge: *challenge_hash.as_bytes(),
            response: response.to_bytes(),
            bit_count,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_scalar() -> Scalar {
        let mut bytes = [0u8; 64];
        rand::fill(&mut bytes);
        Scalar::from_bytes_mod_order_wide(&bytes)
    }

    #[test]
    fn test_commitment_verify() {
        let value = Scalar::from(42u64);
        let blinding = random_scalar();
        let commitment = PedersenCommitment::commit(&value, &blinding);
        assert!(commitment.verify(&value, &blinding));

        let wrong_value = Scalar::from(43u64);
        assert!(!commitment.verify(&wrong_value, &blinding));
    }

    #[test]
    fn test_commitment_homomorphic() {
        let v1 = Scalar::from(10u64);
        let v2 = Scalar::from(20u64);
        let b1 = random_scalar();
        let b2 = random_scalar();

        let c1 = PedersenCommitment::commit(&v1, &b1);
        let c2 = PedersenCommitment::commit(&v2, &b2);
        let c_sum = c1.add(&c2);

        let v_sum = v1 + v2;
        let b_sum = b1 + b2;
        assert!(c_sum.verify(&v_sum, &b_sum));
    }

    #[test]
    fn test_range_proof() {
        let blinding = random_scalar();
        let proof = RangeProof::create(100, &blinding, 8);
        assert!(proof.is_some());

        let overflow = RangeProof::create(300, &blinding, 8);
        assert!(overflow.is_none());
    }
}
