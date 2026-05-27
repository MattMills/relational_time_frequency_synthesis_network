use rtfsn_core::types::NodeId;

use crate::message::ProtocolMessage;

/// Transport abstraction: native UDP, WebSocket, or WebRTC.
///
/// Each DHT layer gets its own transport instance to maintain
/// routing isolation. Traffic analysis on Layer 3 transport
/// cannot reveal Layer 0 topology.
#[async_trait::async_trait]
pub trait Transport: Send + Sync {
    async fn send(&self, target: &NodeId, message: &ProtocolMessage)
        -> Result<(), TransportError>;

    async fn recv(&self) -> Result<(NodeId, ProtocolMessage), TransportError>;

    async fn local_id(&self) -> NodeId;
}

#[derive(Debug, thiserror::Error)]
pub enum TransportError {
    #[error("connection failed: {0}")]
    ConnectionFailed(String),

    #[error("send failed: {0}")]
    SendFailed(String),

    #[error("receive failed: {0}")]
    ReceiveFailed(String),

    #[error("timeout")]
    Timeout,

    #[error("peer unreachable: {0:?}")]
    PeerUnreachable(NodeId),
}
