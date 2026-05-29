use curve25519_dalek::constants::RISTRETTO_BASEPOINT_POINT;
use curve25519_dalek::ristretto::{CompressedRistretto, RistrettoPoint};
use curve25519_dalek::scalar::Scalar;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha512};

use crate::types::{Epoch, NodeId};

fn identity_generator() -> RistrettoPoint {
    let hash = Sha512::digest(b"rtfsn_identity_blinding_generator_v1");
    RistrettoPoint::from_uniform_bytes(
        hash.as_slice().try_into().expect("SHA-512 produces 64 bytes"),
    )
}

/// A blinded identity that cannot be linked back to the original NodeId
/// without knowledge of the blinding factor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlindedIdentity {
    pub commitment: [u8; 32],
    pub epoch: Epoch,
}

/// The secret material needed to prove ownership of a blinded identity.
#[derive(Debug, Clone)]
pub struct BlindingSecret {
    pub node_scalar: Scalar,
    pub blinding_factor: Scalar,
    pub epoch: Epoch,
}

impl BlindingSecret {
    pub fn new(node_id: &NodeId, epoch: Epoch) -> Self {
        let node_scalar = Scalar::from_bytes_mod_order(node_id.0);

        let mut seed = [0u8; 64];
        rand::fill(&mut seed);
        let blinding_factor = Scalar::from_bytes_mod_order_wide(&seed);

        Self {
            node_scalar,
            blinding_factor,
            epoch,
        }
    }

    pub fn blind(&self) -> BlindedIdentity {
        let g = RISTRETTO_BASEPOINT_POINT;
        let h = identity_generator();
        let point = g * self.node_scalar + h * self.blinding_factor;
        BlindedIdentity {
            commitment: point.compress().to_bytes(),
            epoch: self.epoch,
        }
    }

    /// Produce a proof-of-knowledge that we know the opening of the commitment.
    /// Schnorr-like sigma protocol.
    pub fn prove_knowledge(&self) -> BlindingProof {
        let g = RISTRETTO_BASEPOINT_POINT;
        let h = identity_generator();

        let mut k1_bytes = [0u8; 64];
        let mut k2_bytes = [0u8; 64];
        rand::fill(&mut k1_bytes);
        rand::fill(&mut k2_bytes);
        let k1 = Scalar::from_bytes_mod_order_wide(&k1_bytes);
        let k2 = Scalar::from_bytes_mod_order_wide(&k2_bytes);

        let r = g * k1 + h * k2;
        let commitment_point =
            g * self.node_scalar + h * self.blinding_factor;

        let mut hasher = blake3::Hasher::new();
        hasher.update(&commitment_point.compress().to_bytes());
        hasher.update(&r.compress().to_bytes());
        let challenge_hash = hasher.finalize();
        let e = Scalar::from_bytes_mod_order(*challenge_hash.as_bytes());

        let s1 = k1 - e * self.node_scalar;
        let s2 = k2 - e * self.blinding_factor;

        BlindingProof {
            commitment: commitment_point.compress().to_bytes(),
            nonce_commitment: r.compress().to_bytes(),
            s1: s1.to_bytes(),
            s2: s2.to_bytes(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlindingProof {
    pub commitment: [u8; 32],
    pub nonce_commitment: [u8; 32],
    pub s1: [u8; 32],
    pub s2: [u8; 32],
}

impl BlindingProof {
    pub fn verify(&self) -> bool {
        let g = RISTRETTO_BASEPOINT_POINT;
        let h = identity_generator();

        let Some(commitment_point) =
            CompressedRistretto::from_slice(&self.commitment)
                .ok()
                .and_then(|c| c.decompress())
        else {
            return false;
        };

        let Some(r) =
            CompressedRistretto::from_slice(&self.nonce_commitment)
                .ok()
                .and_then(|c| c.decompress())
        else {
            return false;
        };

        let mut hasher = blake3::Hasher::new();
        hasher.update(&self.commitment);
        hasher.update(&self.nonce_commitment);
        let challenge_hash = hasher.finalize();
        let e = Scalar::from_bytes_mod_order(*challenge_hash.as_bytes());

        let s1: Option<Scalar> = Scalar::from_canonical_bytes(self.s1).into();
        let s2: Option<Scalar> = Scalar::from_canonical_bytes(self.s2).into();
        let (Some(s1), Some(s2)) = (s1, s2) else {
            return false;
        };

        let lhs = g * s1 + h * s2 + commitment_point * e;
        lhs == r
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blinding_roundtrip() {
        let node_id = NodeId([42u8; 32]);
        let epoch = Epoch(1);

        let secret1 = BlindingSecret::new(&node_id, epoch);
        let secret2 = BlindingSecret::new(&node_id, epoch);

        let blinded1 = secret1.blind();
        let blinded2 = secret2.blind();

        // Different blinding factors produce different commitments
        assert_ne!(blinded1.commitment, blinded2.commitment);
    }

    #[test]
    fn test_proof_of_knowledge() {
        let node_id = NodeId([7u8; 32]);
        let epoch = Epoch(5);

        let secret = BlindingSecret::new(&node_id, epoch);
        let proof = secret.prove_knowledge();
        assert!(proof.verify());
    }

    #[test]
    fn test_tampered_proof_fails() {
        let node_id = NodeId([7u8; 32]);
        let epoch = Epoch(5);

        let secret = BlindingSecret::new(&node_id, epoch);
        let mut proof = secret.prove_knowledge();
        proof.s1[0] ^= 0xFF;
        assert!(!proof.verify());
    }
}
