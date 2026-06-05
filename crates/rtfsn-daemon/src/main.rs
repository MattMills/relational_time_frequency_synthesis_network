//! rtfsn-daemon: Relational Time-Frequency Synthesis Network daemon
//!
//! BUILD
//!   cargo build --release -p rtfsn-daemon
//!   sudo cp target/release/rtfsn-daemon /usr/local/bin/
//!
//! USAGE — first machine (becomes the bootstrap anchor):
//!   rtfsn-daemon --listen 0.0.0.0:4242
//!
//! USAGE — additional machines (point at any already-running node):
//!   rtfsn-daemon --listen 0.0.0.0:4242 --peers 192.168.1.100:4242
//!   rtfsn-daemon --listen 0.0.0.0:4242 --peers 192.168.1.100:4242,192.168.1.101:4242
//!
//! EPOCH PHASES (default 30s total):
//!   0s–20s  Measure:  exchange NTP-style 4-timestamps with all peers
//!   20s–28s Publish:  broadcast blinded L1 beacon; collect peer beacons
//!   28s–30s Tick:     emit clock tick, print status, advance epoch
//!
//! OUTPUT (printed each epoch):
//!   epoch N | offset ±Xms | drift Yppb | peers Z | geoid K levels | chain N ticks
//!
//! SYSTEMD (save as /etc/systemd/system/rtfsn.service):
//!   [Unit]
//!   Description=RTFSN Clock Daemon
//!   After=network.target
//!   [Service]
//!   ExecStart=/usr/local/bin/rtfsn-daemon --listen 0.0.0.0:4242 --peers 192.168.1.100:4242
//!   Restart=always
//!   RestartSec=5
//!   [Install]
//!   WantedBy=multi-user.target

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use clap::Parser;
use tokio::sync::Mutex;
use tracing::{debug, info, warn};

use rtfsn_core::crypto::blinding::BlindingSecret;
use rtfsn_core::crypto::signatures::NodeKeypair;
use rtfsn_core::geoid::circulation::CirculationManager;
use rtfsn_core::holonomy::chiral::ChiralFrame;
use rtfsn_core::holonomy::solver::HolonomySolver;
use rtfsn_core::holonomy::twist::{TwistIndex, TwistLUT};
use rtfsn_core::layers::layer0::{Layer0Store, LocalSolve, MeasurementRecord};
use rtfsn_core::layers::layer1::Layer1Store;
use rtfsn_core::layers::layer2::ClusterAggregator;
use rtfsn_core::layers::layer3::ClockStream;
use rtfsn_core::layers::valve::{valve_l0_to_l1, valve_l1_to_l2, valve_l2_to_l3};
use rtfsn_core::sync::clock::{ClockState, TimeExchange};
use rtfsn_core::sync::geometry::{CoordinateState, GeometryEngine};
use rtfsn_core::sync::kalman::ClockKalman;
use rtfsn_core::types::{Epoch, NodeId};
use rtfsn_net::message::{PeerStats, ProtocolMessage};
use rtfsn_net::native::UdpTransport;

#[derive(Parser, Debug)]
#[command(name = "rtfsn-daemon")]
#[command(about = "Relational Time-Frequency Synthesis Network daemon")]
struct Args {
    #[arg(short, long, default_value = "0.0.0.0:4242")]
    listen: SocketAddr,

    #[arg(short, long, value_delimiter = ',')]
    peers: Vec<SocketAddr>,

    #[arg(short, long, default_value = "30", help = "Epoch duration in seconds")]
    epoch_duration: u64,

    #[arg(short, long, default_value = "2", help = "Min peers for epoch advance")]
    quorum: usize,

    #[arg(long, help = "Print per-peer RTT and offset details each epoch")]
    verbose: bool,
}

// ─── SolveOutput ────────────────────────────────────────────────────────────

struct SolveOutput {
    offset_nanos: i64,
    uncertainty_nanos: u64,
    drift_ppb: i64,
    peer_count: usize,
    converged: bool,
    iters: u32,
    max_defect_nanos: u64,
}

// ─── NodeState ──────────────────────────────────────────────────────────────

struct NodeState {
    // Identity
    node_id: NodeId,
    keypair: NodeKeypair,
    listen_port: u16,
    verbose: bool,

    // Peers: NodeId → their listen SocketAddr
    peers: HashMap<NodeId, SocketAddr>,

    // In-flight NTP requests: peer_node_id → (t1_nanos, epoch_num)
    pending_t1: HashMap<NodeId, (u64, u64)>,

    // Per-epoch measurements
    twist_lut: TwistLUT,

    // Raw timestamp records: (peer_id, t1, t2, t3, t4)
    raw_exchanges: Vec<(NodeId, u64, u64, u64, u64)>,

    // Layer stores
    l0_store: Layer0Store,
    l1_store: Layer1Store,
    l2_aggregator: ClusterAggregator,
    clock_stream: ClockStream,

    // Geometry
    geometry: GeometryEngine,
    peer_coords: HashMap<NodeId, CoordinateState>,

    // Clock tracking
    clock_state: ClockState,
    kalman: ClockKalman,

    // Geoid / circulation
    circulation: CirculationManager,

    // Last epoch's solved state
    last_offset_nanos: i64,
    last_drift_ppb: i64,
    last_uncertainty_nanos: u64,
    last_peer_count: usize,
    last_solver_converged: bool,
    last_solver_iters: u32,
    last_max_defect_nanos: u64,
    last_rtt_min_ns: u64,
    last_rtt_max_ns: u64,
    last_rtt_mean_ns: u64,
    last_rtt_count: usize,
    last_epoch_start_secs: f64,
    epoch_duration_secs: f64,
    current_epoch: u64,
}

fn now_nanos() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64
}

impl NodeState {
    fn new(
        node_id: NodeId,
        keypair: NodeKeypair,
        listen_port: u16,
        epoch_secs: f64,
        verbose: bool,
    ) -> Self {
        let circulation = CirculationManager::with_levels(Epoch(0), 4);
        Self {
            node_id,
            keypair,
            listen_port,
            verbose,
            peers: HashMap::new(),
            pending_t1: HashMap::new(),
            twist_lut: TwistLUT::new(),
            raw_exchanges: Vec::new(),
            l0_store: Layer0Store::new(node_id),
            l1_store: Layer1Store::new(),
            l2_aggregator: ClusterAggregator::new(),
            clock_stream: ClockStream::new(),
            geometry: GeometryEngine::new(),
            peer_coords: HashMap::new(),
            clock_state: ClockState::new(),
            kalman: ClockKalman::new(),
            circulation,
            last_offset_nanos: 0,
            last_drift_ppb: 0,
            last_uncertainty_nanos: 1_000_000_000,
            last_peer_count: 0,
            last_solver_converged: false,
            last_solver_iters: 0,
            last_max_defect_nanos: 0,
            last_rtt_min_ns: 0,
            last_rtt_max_ns: 0,
            last_rtt_mean_ns: 0,
            last_rtt_count: 0,
            last_epoch_start_secs: 0.0,
            epoch_duration_secs: epoch_secs,
            current_epoch: 0,
        }
    }

    fn solve_and_advance(&mut self, epoch: Epoch) -> SolveOutput {
        let current_secs =
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs_f64();
        let dt = if self.last_epoch_start_secs < 1.0 {
            self.epoch_duration_secs
        } else {
            (current_secs - self.last_epoch_start_secs).max(1.0)
        };

        // Build solver: add self with last known offset as seed
        let mut solver = HolonomySolver::new();
        solver.add_node(
            self.node_id,
            ChiralFrame::new(self.last_offset_nanos, self.last_drift_ppb, 90),
        );

        // Add all known peers at 0 seed (solver discovers their offsets)
        for peer_id in self.peers.keys().copied() {
            solver.add_node(peer_id, ChiralFrame::new(0, 0, 50));
        }

        // Add all twist measurements from the LUT into the solver
        let edges: Vec<(NodeId, NodeId)> = self
            .twist_lut
            .neighbors(self.node_id)
            .into_iter()
            .map(|n| (self.node_id, n))
            .collect();

        for (a, b) in &edges {
            if let Some(twist) = self.twist_lut.latest(*a, *b) {
                solver.add_measurement(*a, *b, twist.clone());
            }
        }

        // Run solver
        let result = solver.solve(500, 1000);

        // Extract our solved offset
        let offset_nanos = solver
            .nodes
            .get(&self.node_id)
            .map(|f| f.offset_nanos)
            .unwrap_or(self.last_offset_nanos);

        let drift_ppb = solver
            .nodes
            .get(&self.node_id)
            .map(|f| f.drift_ppb)
            .unwrap_or(self.last_drift_ppb);

        // Compute RTT stats from twist_lut edges involving self
        let mut rtt_values: Vec<u64> = Vec::new();
        for peer_id in self.twist_lut.neighbors(self.node_id) {
            if let Some(twist) = self.twist_lut.latest(self.node_id, peer_id) {
                rtt_values.push(twist.rtt_nanos);
            }
        }
        let (rtt_min, rtt_max, rtt_mean, rtt_count) = if rtt_values.is_empty() {
            (0u64, 0u64, 0u64, 0usize)
        } else {
            let min = *rtt_values.iter().min().unwrap();
            let max = *rtt_values.iter().max().unwrap();
            let mean = rtt_values.iter().sum::<u64>() / rtt_values.len() as u64;
            (min, max, mean, rtt_values.len())
        };

        // Update geometry: observe each peer that has a twist
        for peer_id in self.twist_lut.neighbors(self.node_id) {
            if let Some(twist) = self.twist_lut.latest(self.node_id, peer_id) {
                let rtt_secs = twist.rtt_nanos as f64 / 1e9;
                let peer_coords = self
                    .peer_coords
                    .entry(peer_id)
                    .or_insert_with(CoordinateState::new)
                    .clone();
                self.geometry.observe_peer(peer_id, peer_coords, rtt_secs);
            }
        }

        // Update Kalman filter
        let offset_secs = offset_nanos as f64 / 1e9;
        let var = if rtt_mean > 0 {
            let rtt_secs = rtt_mean as f64 / 1e9;
            rtt_secs * rtt_secs * 0.25
        } else {
            1e-6
        };
        self.kalman.step(dt, offset_secs, var);

        // Build Layer 0 MeasurementRecord from raw exchanges
        let peer_count = self.raw_exchanges.len();
        let measurements: Vec<MeasurementRecord> = self
            .raw_exchanges
            .iter()
            .map(|(peer_id, t1, t2, t3, t4)| {
                let rtt_secs = (*t4 as f64 - *t1 as f64) / 1e9
                    - (*t3 as f64 - *t2 as f64) / 1e9;
                let offset_s = ((*t2 as f64 - *t1 as f64)
                    + (*t3 as f64 - *t4 as f64))
                    / 2.0;
                // Construct synthetic TimeExchange that produces the same offset/RTT
                let exchange = TimeExchange::new(
                    0.0,
                    rtt_secs / 2.0 + offset_s,
                    rtt_secs / 2.0,
                    rtt_secs,
                );
                MeasurementRecord {
                    peer: *peer_id,
                    exchange,
                    measured_rtt: rtt_secs,
                    local_time: current_secs,
                }
            })
            .collect();

        // Also update clock_state with each exchange
        for record in &measurements {
            self.clock_state.add_exchange(&record.exchange, current_secs);
        }

        let solve = LocalSolve {
            coordinates: self.geometry.local.clone(),
            clock_offset: offset_secs,
            clock_drift: self.kalman.drift_rate(),
            uncertainty: self.kalman.uncertainty(),
            peer_count,
        };

        self.l0_store.record_epoch(epoch, measurements, solve);

        let uncertainty_nanos = (self.kalman.uncertainty() * 1e9) as u64;

        // Update last_ fields
        self.last_offset_nanos = offset_nanos;
        self.last_drift_ppb = drift_ppb;
        self.last_uncertainty_nanos = uncertainty_nanos;
        self.last_peer_count = peer_count;
        self.last_solver_converged = result.converged;
        self.last_solver_iters = result.iterations;
        self.last_max_defect_nanos = result.max_defect_nanos;
        self.last_rtt_min_ns = rtt_min;
        self.last_rtt_max_ns = rtt_max;
        self.last_rtt_mean_ns = rtt_mean;
        self.last_rtt_count = rtt_count;
        self.last_epoch_start_secs = current_secs;

        // Clear per-epoch raw exchanges for the next epoch
        self.raw_exchanges.clear();

        SolveOutput {
            offset_nanos,
            uncertainty_nanos,
            drift_ppb,
            peer_count,
            converged: result.converged,
            iters: result.iterations,
            max_defect_nanos: result.max_defect_nanos,
        }
    }

    fn build_and_publish_beacon(&mut self, epoch: Epoch) -> Option<Vec<u8>> {
        let blinding = BlindingSecret::new(&self.node_id, epoch);
        let beacon = valve_l0_to_l1(&self.l0_store, epoch, &blinding)?;
        self.l1_store.insert(beacon.clone());
        serde_json::to_vec(&beacon).ok()
    }

    fn run_geoid_and_emit_tick(&mut self, epoch: Epoch) {
        let beacon_refs: Vec<&rtfsn_core::layers::layer1::Layer1Beacon> =
            self.l1_store.get_epoch(epoch);

        self.circulation.process_upward(&beacon_refs, epoch);

        valve_l1_to_l2(&self.l1_store, &mut self.l2_aggregator, epoch, 0.01);

        valve_l2_to_l3(
            &self.l2_aggregator,
            &mut self.clock_stream,
            epoch,
            now_nanos(),
            1,
        );
    }

    fn build_stats_response(&self) -> ProtocolMessage {
        let peers: Vec<PeerStats> = self
            .twist_lut
            .neighbors(self.node_id)
            .into_iter()
            .filter_map(|peer_id| {
                let twist = self.twist_lut.latest(self.node_id, peer_id)?;
                let profile = self.twist_lut.profile(self.node_id, peer_id);
                let addr = self
                    .peers
                    .get(&peer_id)
                    .map(|a| a.to_string())
                    .unwrap_or_else(|| "unknown".to_string());
                let rtt_jitter_ns = profile
                    .as_ref()
                    .map(|p| (p.variance_nanos as f64).sqrt() as u64)
                    .unwrap_or(0);
                let trend_nanos_per_epoch =
                    profile.as_ref().map(|p| p.trend_nanos_per_epoch).unwrap_or(0);
                let sample_count =
                    profile.as_ref().map(|p| p.sample_count).unwrap_or(1);
                Some(PeerStats {
                    peer_id,
                    peer_addr: addr,
                    rtt_nanos: twist.rtt_nanos,
                    rtt_jitter_ns,
                    offset_nanos: twist.offset_nanos,
                    asymmetry_nanos: twist.asymmetry_nanos,
                    quality: twist.quality,
                    epoch_measured: twist.epoch_measured,
                    trend_nanos_per_epoch,
                    sample_count,
                })
            })
            .collect();

        // Aggregate jitter: std dev of per-peer RTT values
        let rtt_jitter_ns = if peers.len() > 1 {
            let mean = self.last_rtt_mean_ns;
            let var = peers
                .iter()
                .map(|p| {
                    let d = p.rtt_nanos as i64 - mean as i64;
                    (d * d) as u64
                })
                .sum::<u64>()
                / peers.len() as u64;
            (var as f64).sqrt() as u64
        } else {
            0
        };

        let geoid_region_counts: Vec<u32> = self
            .circulation
            .geoid
            .layers
            .iter()
            .map(|l| l.regions.len() as u32)
            .collect();

        ProtocolMessage::StatsResponse {
            node_id: self.node_id,
            epoch: self.current_epoch,
            offset_nanos: self.last_offset_nanos,
            uncertainty_nanos: self.last_uncertainty_nanos,
            drift_ppb: self.last_drift_ppb,
            solver_converged: self.last_solver_converged,
            solver_iters: self.last_solver_iters,
            max_defect_nanos: self.last_max_defect_nanos,
            rtt_min_ns: self.last_rtt_min_ns,
            rtt_mean_ns: self.last_rtt_mean_ns,
            rtt_max_ns: self.last_rtt_max_ns,
            rtt_jitter_ns,
            geoid_depth: self.circulation.geoid.depth(),
            geoid_region_counts,
            chain_length: self.clock_stream.len() as u64,
            chain_valid: self.clock_stream.verify_all(),
            peers,
        }
    }

    fn print_status(&self, epoch: Epoch) {
        let offset_ms = self.last_offset_nanos as f64 / 1_000_000.0;
        let uncertainty_ms = self.last_uncertainty_nanos as f64 / 1_000_000.0;
        let drift_ppb = self.last_drift_ppb;

        let solver_state = if self.last_solver_converged {
            "converged"
        } else {
            "timeout"
        };

        let geoid_depth = self.circulation.geoid.depth();
        let region_counts: Vec<String> = self
            .circulation
            .geoid
            .layers
            .iter()
            .map(|l| l.regions.len().to_string())
            .collect();
        let region_str = if region_counts.is_empty() {
            "none".to_string()
        } else {
            region_counts.join(" -> ")
        };

        let chain_len = self.clock_stream.len();
        let chain_valid = if self.clock_stream.verify_all() {
            "[OK]"
        } else {
            "[!!]"
        };

        let rtt_min_ms = self.last_rtt_min_ns as f64 / 1_000_000.0;
        let rtt_mean_ms = self.last_rtt_mean_ns as f64 / 1_000_000.0;
        let rtt_max_ms = self.last_rtt_max_ns as f64 / 1_000_000.0;

        let node_hex: String = self.node_id.0[..4]
            .iter()
            .map(|b| format!("{:02x}", b))
            .collect();

        println!("══════════════════════════════════════════════════════");
        println!(" epoch {} | node {}", epoch.0, node_hex);
        println!("──────────────────────────────────────────────────────");
        println!(
            "  offset  {:+.3} ms  ±{:.3} ms   drift {:+.1} ppb",
            offset_ms, uncertainty_ms, drift_ppb
        );
        println!(
            "  peers {}   solver {} in {} iters",
            self.last_peer_count, solver_state, self.last_solver_iters
        );
        println!("  geoid {} levels ({})", geoid_depth, region_str);
        println!("  chain {} ticks  {}", chain_len, chain_valid);
        println!(
            "  RTT: min={:.1}ms mean={:.1}ms max={:.1}ms",
            rtt_min_ms, rtt_mean_ms, rtt_max_ms
        );
        println!("══════════════════════════════════════════════════════");
    }
}

// ─── recv_loop ──────────────────────────────────────────────────────────────

async fn recv_loop(state: Arc<Mutex<NodeState>>, transport: Arc<UdpTransport>) {
    loop {
        match transport.recv().await {
            Ok((from_addr, msg)) => {
                handle_message(Arc::clone(&state), Arc::clone(&transport), from_addr, msg)
                    .await
            }
            Err(e) => warn!("recv: {e}"),
        }
    }
}

async fn handle_message(
    state: Arc<Mutex<NodeState>>,
    transport: Arc<UdpTransport>,
    from_addr: SocketAddr,
    msg: ProtocolMessage,
) {
    match msg {
        ProtocolMessage::PeerAnnounce { node_id, listen_port } => {
            let listen_addr = SocketAddr::new(from_addr.ip(), listen_port);
            let (our_node_id, our_port) = {
                let mut s = state.lock().await;
                let inserted = s.peers.insert(node_id, listen_addr).is_none();
                if inserted {
                    debug!("new peer: {:?} at {}", node_id, listen_addr);
                }
                (s.node_id, s.listen_port)
            };
            // Send our own PeerAnnounce back (lock already released)
            transport
                .send_to_addr(
                    listen_addr,
                    &ProtocolMessage::PeerAnnounce {
                        node_id: our_node_id,
                        listen_port: our_port,
                    },
                )
                .await
                .ok();
        }

        ProtocolMessage::TimeRequest { sender, t1_nanos, epoch } => {
            // Record t2 IMMEDIATELY before lock
            let t2 = now_nanos();
            {
                let mut s = state.lock().await;
                s.peers.entry(sender).or_insert_with(|| {
                    SocketAddr::new(from_addr.ip(), from_addr.port())
                });
            }
            let t3 = now_nanos();
            let our_node_id = state.lock().await.node_id;
            transport
                .send_to_addr(
                    from_addr,
                    &ProtocolMessage::TimeResponse {
                        sender: our_node_id,
                        t1_nanos,
                        t2_nanos: t2,
                        t3_nanos: t3,
                        epoch,
                    },
                )
                .await
                .ok();
        }

        ProtocolMessage::TimeResponse {
            sender,
            t1_nanos,
            t2_nanos,
            t3_nanos,
            epoch,
        } => {
            let t4 = now_nanos();
            let mut s = state.lock().await;

            // Check pending_t1 for a matching entry
            let valid = s
                .pending_t1
                .get(&sender)
                .map(|(stored_t1, stored_epoch)| {
                    *stored_t1 == t1_nanos && *stored_epoch == epoch.0
                })
                .unwrap_or(false);

            if !valid {
                debug!("ignoring unexpected TimeResponse from {:?}", sender);
                return;
            }

            s.pending_t1.remove(&sender);

            // Register peer addr if not already known
            s.peers.entry(sender).or_insert_with(|| {
                SocketAddr::new(from_addr.ip(), from_addr.port())
            });

            // Compute TwistIndex from the 4 timestamps
            let rtt = (t4.wrapping_sub(t1_nanos))
                .wrapping_sub(t3_nanos.wrapping_sub(t2_nanos));
            let offset = ((t2_nanos as i128 - t1_nanos as i128)
                + (t3_nanos as i128 - t4 as i128))
                / 2;
            let forward = t2_nanos as i128 - t1_nanos as i128;
            let backward = t4 as i128 - t3_nanos as i128;
            let asymmetry = forward - backward;

            let twist = TwistIndex {
                rtt_nanos: rtt,
                offset_nanos: offset as i64,
                asymmetry_nanos: asymmetry as i64,
                quality: 90,
                epoch_measured: epoch.0,
            };

            let local_id = s.node_id;
            s.twist_lut.insert(local_id, sender, twist);
            s.raw_exchanges.push((sender, t1_nanos, t2_nanos, t3_nanos, t4));
        }

        ProtocolMessage::Beacon { data, epoch } => {
            if let Ok(beacon) =
                serde_json::from_slice::<rtfsn_core::layers::layer1::Layer1Beacon>(&data)
            {
                let mut s = state.lock().await;
                let inserted = s.l1_store.insert(beacon);
                debug!("received beacon epoch={} inserted={}", epoch.0, inserted);
            } else {
                debug!("failed to deserialize beacon from {}", from_addr);
            }
        }

        ProtocolMessage::Ping { nonce } => {
            transport
                .send_to_addr(from_addr, &ProtocolMessage::Pong { nonce })
                .await
                .ok();
        }

        ProtocolMessage::StatsRequest {} => {
            let response = state.lock().await.build_stats_response();
            transport.send_to_addr(from_addr, &response).await.ok();
        }

        other => {
            debug!(
                "ignoring message variant {:?} from {}",
                std::mem::discriminant(&other),
                from_addr
            );
        }
    }
}

// ─── epoch_loop ─────────────────────────────────────────────────────────────

async fn epoch_loop(
    state: Arc<Mutex<NodeState>>,
    transport: Arc<UdpTransport>,
    args: Args,
) {
    let epoch_nanos = args.epoch_duration * 1_000_000_000;
    let measure_nanos = epoch_nanos * 2 / 3;
    let publish_nanos = epoch_nanos / 6;
    // tick phase = remaining ~1/6

    // Initial peer bootstrap: send PeerAnnounce to all bootstrap addrs
    {
        let (node_id, listen_port) = {
            let s = state.lock().await;
            (s.node_id, s.listen_port)
        };
        for addr in &args.peers {
            transport
                .send_to_addr(
                    *addr,
                    &ProtocolMessage::PeerAnnounce {
                        node_id,
                        listen_port,
                    },
                )
                .await
                .ok();
        }
    }

    let mut epoch_num: u64 = 0;

    loop {
        let epoch = Epoch(epoch_num);
        state.lock().await.current_epoch = epoch_num;

        // ── MEASURE PHASE ──────────────────────────────────────────────────
        {
            let (node_id, listen_port, peers) = {
                let s = state.lock().await;
                (s.node_id, s.listen_port, s.peers.clone())
            };

            for (peer_id, peer_addr) in &peers {
                let t1 = now_nanos();
                {
                    let mut s = state.lock().await;
                    s.pending_t1.insert(*peer_id, (t1, epoch_num));
                }
                transport
                    .send_to_addr(
                        *peer_addr,
                        &ProtocolMessage::TimeRequest {
                            sender: node_id,
                            t1_nanos: t1,
                            epoch,
                        },
                    )
                    .await
                    .ok();
                // Re-announce ourselves in case peer restarted
                transport
                    .send_to_addr(
                        *peer_addr,
                        &ProtocolMessage::PeerAnnounce {
                            node_id,
                            listen_port,
                        },
                    )
                    .await
                    .ok();
            }
        }

        tokio::time::sleep(Duration::from_nanos(measure_nanos)).await;

        // ── SOLVE PHASE (synchronous, brief) ───────────────────────────────
        let beacon_bytes = {
            let mut s = state.lock().await;
            let _solution = s.solve_and_advance(epoch);
            s.build_and_publish_beacon(epoch)
        };

        // ── PUBLISH PHASE ──────────────────────────────────────────────────
        if let Some(bytes) = beacon_bytes {
            let peers = state.lock().await.peers.clone();
            for (_, peer_addr) in &peers {
                transport
                    .send_to_addr(
                        *peer_addr,
                        &ProtocolMessage::Beacon {
                            data: bytes.clone(),
                            epoch,
                        },
                    )
                    .await
                    .ok();
            }
        }

        tokio::time::sleep(Duration::from_nanos(publish_nanos)).await;

        // ── TICK PHASE ─────────────────────────────────────────────────────
        {
            let mut s = state.lock().await;
            s.run_geoid_and_emit_tick(epoch);
            s.print_status(epoch);
            // Clean up stale pending_t1 entries from this epoch
            s.pending_t1.retain(|_, (_, ep)| *ep != epoch_num);
        }

        epoch_num += 1;
    }
}

// ─── main ───────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "rtfsn=info".parse().unwrap()),
        )
        .init();

    let args = Args::parse();

    let keypair = NodeKeypair::generate();
    let node_id = keypair.node_id;

    let node_hex: String = node_id.0[..4]
        .iter()
        .map(|b| format!("{:02x}", b))
        .collect();

    info!("══════════════════════════════════════════════════════");
    info!(" RTFSN daemon starting");
    info!("  node_id   : {}", node_hex);
    info!("  listen    : {}", args.listen);
    info!(
        "  peers     : {}",
        if args.peers.is_empty() {
            "(none — bootstrap anchor)".to_string()
        } else {
            args.peers
                .iter()
                .map(|a| a.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        }
    );
    info!(
        "  epoch     : {}s  quorum: {}  verbose: {}",
        args.epoch_duration, args.quorum, args.verbose
    );
    info!("══════════════════════════════════════════════════════");

    let listen_port = args.listen.port();
    let verbose = args.verbose;
    let epoch_secs = args.epoch_duration as f64;

    let node_state = NodeState::new(node_id, keypair, listen_port, epoch_secs, verbose);
    let state = Arc::new(Mutex::new(node_state));

    let transport = Arc::new(UdpTransport::bind(args.listen, node_id).await?);
    info!("listening on {}", transport.local_addr()?);

    // Spawn receive task
    let recv_state = Arc::clone(&state);
    let recv_transport = Arc::clone(&transport);
    tokio::spawn(async move {
        recv_loop(recv_state, recv_transport).await;
    });

    // Run epoch loop on main task (blocks forever)
    epoch_loop(state, transport, args).await;

    Ok(())
}
