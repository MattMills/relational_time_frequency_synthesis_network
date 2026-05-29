use serde::{Deserialize, Serialize};

/// Simplified iterative-hashing VDF for prototyping.
/// Production would use Wesolowski or Pietrzak constructions over groups
/// of unknown order.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VdfProof {
    pub input: [u8; 32],
    pub output: [u8; 32],
    pub iterations: u64,
}

impl VdfProof {
    pub fn evaluate(input: &[u8; 32], iterations: u64) -> Self {
        let mut current = *input;
        for _ in 0..iterations {
            current = *blake3::hash(&current).as_bytes();
        }
        Self {
            input: *input,
            output: current,
            iterations,
        }
    }

    pub fn verify(&self) -> bool {
        let mut current = self.input;
        for _ in 0..self.iterations {
            current = *blake3::hash(&current).as_bytes();
        }
        current == self.output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vdf_evaluate_verify() {
        let input = *blake3::hash(b"epoch_seed_42").as_bytes();
        let proof = VdfProof::evaluate(&input, 1000);
        assert!(proof.verify());
    }

    #[test]
    fn test_vdf_wrong_output() {
        let input = *blake3::hash(b"epoch_seed_42").as_bytes();
        let mut proof = VdfProof::evaluate(&input, 1000);
        proof.output[0] ^= 1;
        assert!(!proof.verify());
    }
}
