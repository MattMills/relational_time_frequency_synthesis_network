use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;

use rtfsn_core::types::NodeId;
use tokio::net::UdpSocket;
use tokio::sync::Mutex;

use crate::message::ProtocolMessage;
use crate::transport::TransportError;

/// Native UDP transport for daemon mode.
pub struct UdpTransport {
    socket: Arc<UdpSocket>,
    local_node_id: NodeId,
    peer_addrs: Arc<Mutex<HashMap<NodeId, SocketAddr>>>,
}

impl UdpTransport {
    pub async fn bind(
        addr: SocketAddr,
        local_node_id: NodeId,
    ) -> Result<Self, TransportError> {
        let socket = UdpSocket::bind(addr)
            .await
            .map_err(|e| TransportError::ConnectionFailed(e.to_string()))?;

        Ok(Self {
            socket: Arc::new(socket),
            local_node_id,
            peer_addrs: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    pub async fn register_peer(
        &self,
        node_id: NodeId,
        addr: SocketAddr,
    ) {
        self.peer_addrs.lock().await.insert(node_id, addr);
    }

    pub async fn send(
        &self,
        target: &NodeId,
        message: &ProtocolMessage,
    ) -> Result<(), TransportError> {
        let addrs = self.peer_addrs.lock().await;
        let addr = addrs
            .get(target)
            .ok_or(TransportError::PeerUnreachable(*target))?;

        let data = message
            .serialize()
            .map_err(|e| TransportError::SendFailed(e.to_string()))?;

        self.socket
            .send_to(&data, addr)
            .await
            .map_err(|e| TransportError::SendFailed(e.to_string()))?;

        Ok(())
    }

    pub async fn recv(
        &self,
    ) -> Result<(SocketAddr, ProtocolMessage), TransportError> {
        let mut buf = vec![0u8; 65536];
        let (len, addr) = self
            .socket
            .recv_from(&mut buf)
            .await
            .map_err(|e| TransportError::ReceiveFailed(e.to_string()))?;

        let message = ProtocolMessage::deserialize(&buf[..len])
            .map_err(|e| TransportError::ReceiveFailed(e.to_string()))?;

        Ok((addr, message))
    }

    pub fn local_node_id(&self) -> NodeId {
        self.local_node_id
    }

    pub fn local_addr(&self) -> Result<SocketAddr, TransportError> {
        self.socket
            .local_addr()
            .map_err(|e| TransportError::ConnectionFailed(e.to_string()))
    }

    pub async fn send_to_addr(
        &self,
        addr: SocketAddr,
        message: &ProtocolMessage,
    ) -> Result<(), TransportError> {
        let data = message
            .serialize()
            .map_err(|e| TransportError::SendFailed(e.to_string()))?;
        self.socket
            .send_to(&data, addr)
            .await
            .map_err(|e| TransportError::SendFailed(e.to_string()))?;
        Ok(())
    }
}
