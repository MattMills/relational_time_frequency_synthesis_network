use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};

use crate::types::NodeId;

pub struct NodeKeypair {
    signing_key: SigningKey,
    pub verifying_key: VerifyingKey,
    pub node_id: NodeId,
}

impl NodeKeypair {
    pub fn generate() -> Self {
        let mut csprng_bytes = [0u8; 32];
        rand::fill(&mut csprng_bytes);
        let signing_key = SigningKey::from_bytes(&csprng_bytes);
        let verifying_key = signing_key.verifying_key();
        let node_id = NodeId::from_public_key(&verifying_key);

        Self {
            signing_key,
            verifying_key,
            node_id,
        }
    }

    pub fn sign(&self, message: &[u8]) -> SignedMessage {
        let signature = self.signing_key.sign(message);
        SignedMessage {
            payload: message.to_vec(),
            signature: signature.to_bytes().to_vec(),
            signer: self.verifying_key.to_bytes(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignedMessage {
    pub payload: Vec<u8>,
    pub signature: Vec<u8>,
    pub signer: [u8; 32],
}

impl SignedMessage {
    pub fn verify(&self) -> bool {
        let Ok(verifying_key) = VerifyingKey::from_bytes(&self.signer) else {
            return false;
        };
        let sig_bytes: [u8; 64] = match self.signature.as_slice().try_into() {
            Ok(b) => b,
            Err(_) => return false,
        };
        let signature = Signature::from_bytes(&sig_bytes);
        verifying_key.verify(&self.payload, &signature).is_ok()
    }

    pub fn signer_node_id(&self) -> NodeId {
        let Ok(vk) = VerifyingKey::from_bytes(&self.signer) else {
            return NodeId([0u8; 32]);
        };
        NodeId::from_public_key(&vk)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sign_verify() {
        let kp = NodeKeypair::generate();
        let msg = b"hello network time";
        let signed = kp.sign(msg);
        assert!(signed.verify());
        assert_eq!(signed.signer_node_id(), kp.node_id);
    }

    #[test]
    fn test_tampered_message() {
        let kp = NodeKeypair::generate();
        let mut signed = kp.sign(b"original");
        signed.payload = b"tampered".to_vec();
        assert!(!signed.verify());
    }
}
