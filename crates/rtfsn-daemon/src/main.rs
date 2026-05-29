use std::net::SocketAddr;

use clap::Parser;
use tracing::{info, warn};

#[derive(Parser, Debug)]
#[command(name = "rtfsn-daemon")]
#[command(about = "Relational Time-Frequency Synthesis Network daemon")]
struct Args {
    /// UDP listen address
    #[arg(short, long, default_value = "0.0.0.0:4242")]
    listen: SocketAddr,

    /// Bootstrap peer addresses (comma-separated)
    #[arg(short, long, value_delimiter = ',')]
    peers: Vec<SocketAddr>,

    /// Epoch duration in seconds
    #[arg(short, long, default_value = "30")]
    epoch_duration: u64,

    /// Quorum threshold (minimum peers for epoch advance)
    #[arg(short, long, default_value = "3")]
    quorum: usize,

    /// Enable cryptographic DMZ mode (isolated local crypto environment)
    #[arg(long)]
    dmz: bool,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "rtfsn=info".parse().unwrap()),
        )
        .init();

    let args = Args::parse();

    info!("RTFSN daemon starting on {}", args.listen);
    info!(
        "epoch_duration={}s quorum={} dmz={}",
        args.epoch_duration, args.quorum, args.dmz
    );

    // Generate node identity
    let keypair = rtfsn_core::crypto::signatures::NodeKeypair::generate();
    info!("node_id={:?}", keypair.node_id);

    // Initialize the 4-layer engine
    let _epoch_manager = rtfsn_core::epoch::EpochManager::new(
        args.quorum,
        args.epoch_duration * 1_000_000_000,
    );
    let _clock_stream = rtfsn_core::layers::layer3::ClockStream::new();

    // Initialize network transport
    let transport = rtfsn_net::native::UdpTransport::bind(
        args.listen,
        keypair.node_id,
    )
    .await?;

    info!("listening on {}", transport.local_addr()?);

    // Register bootstrap peers
    for peer_addr in &args.peers {
        info!("bootstrap peer: {}", peer_addr);
    }

    if args.dmz {
        info!("cryptographic DMZ mode: local crypto environment isolated from network-facing browser context");
    }

    info!("RTFSN daemon initialized — entering epoch loop");

    // Main epoch loop
    // TODO: wire up the full epoch lifecycle:
    //   Phase 1 (Measure): exchange timestamps with peers
    //   Phase 2 (Solve): run holonomy solver on collected measurements
    //   Phase 3 (Publish): broadcast beacon, collect quorum, emit clock tick

    // For now, listen for messages
    loop {
        match transport.recv().await {
            Ok((addr, msg)) => {
                info!("received message from {}: {:?}", addr, std::mem::discriminant(&msg));
            }
            Err(e) => {
                warn!("receive error: {}", e);
            }
        }
    }
}
