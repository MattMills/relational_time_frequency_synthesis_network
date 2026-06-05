use rtfsn_core::types::{Epoch, NodeId};
use serde::{Deserialize, Serialize};

/// Per-peer statistics carried in a StatsResponse.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerStats {
    pub peer_id: NodeId,
    /// Stringified SocketAddr of the peer.
    pub peer_addr: String,
    /// Most recent measured round-trip time (nanoseconds).
    pub rtt_nanos: u64,
    /// RTT standard deviation across measurement history (nanoseconds).
    pub rtt_jitter_ns: u64,
    /// Most recent clock offset measurement (nanoseconds).
    pub offset_nanos: i64,
    /// Asymmetry between forward and backward one-way delays (nanoseconds).
    pub asymmetry_nanos: i64,
    pub quality: u32,
    pub epoch_measured: u64,
    /// Linear trend of RTT over epochs (nanoseconds per epoch; negative = improving).
    pub trend_nanos_per_epoch: i64,
    pub sample_count: u32,
}

/// Protocol messages exchanged between nodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProtocolMessage {
    /// NTP-like time exchange: step 1 (client → server)
    TimeRequest {
        sender: NodeId,
        t1_nanos: u64,
        epoch: Epoch,
    },

    /// NTP-like time exchange: step 2 (server → client)
    TimeResponse {
        sender: NodeId,
        t1_nanos: u64,
        t2_nanos: u64,
        t3_nanos: u64,
        epoch: Epoch,
    },

    /// Layer 1 beacon broadcast
    Beacon {
        data: Vec<u8>,
        epoch: Epoch,
    },

    /// Layer 2 cluster commitment
    ClusterCommitment {
        data: Vec<u8>,
        epoch: Epoch,
    },

    /// Layer 3 clock tick broadcast
    ClockTick {
        data: Vec<u8>,
        epoch: Epoch,
    },

    /// DHT routing: find node
    FindNode {
        target: [u8; 32],
        layer: u8,
    },

    /// DHT routing: found nodes
    FoundNodes {
        nodes: Vec<(NodeId, Vec<u8>)>,
        layer: u8,
    },

    /// DHT storage: put
    Store {
        key: [u8; 32],
        value: Vec<u8>,
        layer: u8,
    },

    /// DHT storage: get
    Retrieve {
        key: [u8; 32],
        layer: u8,
    },

    /// DHT storage: get response
    RetrieveResponse {
        key: [u8; 32],
        value: Option<Vec<u8>>,
        layer: u8,
    },

    /// Ping/pong for liveness
    Ping { nonce: u64 },
    Pong { nonce: u64 },

    /// Peer discovery. Sent by new nodes to bootstrap peers.
    /// listen_port lets the receiver know where to send future messages.
    PeerAnnounce {
        node_id: NodeId,
        listen_port: u16,
    },

    /// Stats query request from rtfsn-query tool.
    StatsRequest {},

    /// Stats query response.
    StatsResponse {
        node_id: NodeId,
        epoch: u64,
        offset_nanos: i64,
        uncertainty_nanos: u64,
        drift_ppb: i64,
        solver_converged: bool,
        solver_iters: u32,
        max_defect_nanos: u64,
        rtt_min_ns: u64,
        rtt_mean_ns: u64,
        rtt_max_ns: u64,
        /// Standard deviation of per-peer RTTs (nanoseconds).
        rtt_jitter_ns: u64,
        geoid_depth: u8,
        geoid_region_counts: Vec<u32>,
        chain_length: u64,
        chain_valid: bool,
        peers: Vec<PeerStats>,
    },
}

impl ProtocolMessage {
    pub fn serialize(&self) -> Result<Vec<u8>, serde_json::Error> {
        serde_json::to_vec(self)
    }

    pub fn deserialize(data: &[u8]) -> Result<Self, serde_json::Error> {
        serde_json::from_slice(data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_roundtrip() {
        let msg = ProtocolMessage::TimeRequest {
            sender: NodeId([42u8; 32]),
            t1_nanos: 1_000_000_000,
            epoch: Epoch(5),
        };

        let bytes = msg.serialize().unwrap();
        let decoded = ProtocolMessage::deserialize(&bytes).unwrap();

        match decoded {
            ProtocolMessage::TimeRequest {
                sender,
                t1_nanos,
                epoch,
            } => {
                assert_eq!(sender, NodeId([42u8; 32]));
                assert_eq!(t1_nanos, 1_000_000_000);
                assert_eq!(epoch, Epoch(5));
            }
            _ => panic!("wrong message type"),
        }
    }
}
