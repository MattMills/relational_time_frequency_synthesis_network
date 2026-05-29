use std::collections::HashMap;

use rand::Rng;
use rand_distr::{Distribution, Normal};

use rtfsn_core::holonomy::chiral::ChiralFrame;
use rtfsn_core::holonomy::samr::SAMRState;
use rtfsn_core::holonomy::solver::HolonomySolver;
use rtfsn_core::holonomy::temporal_mirror::EpochBoundarySolver;
use rtfsn_core::holonomy::twist::TwistIndex;
use rtfsn_core::layers::layer3::ClockStream;
use rtfsn_core::sync::kalman::ClockKalman;
use rtfsn_core::sync::outlier::{marzullo, TimeInterval};
use rtfsn_core::types::NodeId;

// ---------------------------------------------------------------------------
// Simulated node
// ---------------------------------------------------------------------------

struct SimNode {
    id: NodeId,
    true_offset_nanos: i64,
    true_drift_ppb: i64,
    kalman: ClockKalman,
    samr: SAMRState,
    peers: Vec<usize>,
    last_solved_offset_nanos: i64,
}

impl SimNode {
    fn local_time_nanos(&self, real_time_nanos: i64) -> i64 {
        real_time_nanos
            + self.true_offset_nanos
            + (self.true_drift_ppb * real_time_nanos) / 1_000_000_000
    }
}

// ---------------------------------------------------------------------------
// Simulated network
// ---------------------------------------------------------------------------

struct SimNetwork {
    nodes: Vec<SimNode>,
    base_latencies: Vec<Vec<u64>>,
    jitter_stddev_nanos: f64,
}

impl SimNetwork {
    fn simulate_exchange(
        &self,
        rng: &mut impl Rng,
        a: usize,
        b: usize,
        real_time_nanos: i64,
    ) -> TwistIndex {
        let jitter = Normal::new(0.0, self.jitter_stddev_nanos).unwrap();
        let base_lat = self.base_latencies[a][b];

        let forward_delay = base_lat as f64 + jitter.sample(rng).abs();
        let backward_delay = base_lat as f64 + jitter.sample(rng).abs();

        let t1 = self.nodes[a].local_time_nanos(real_time_nanos) as u64;
        let t2 = (self.nodes[b].local_time_nanos(real_time_nanos) as f64
            + forward_delay) as u64;
        let processing = 100_000; // 100μs
        let t3 = t2 + processing;
        let t4 = (self.nodes[a].local_time_nanos(real_time_nanos) as f64
            + forward_delay
            + backward_delay
            + processing as f64) as u64;

        TwistIndex::from_exchange(t1, t2, t3, t4)
    }
}

// ---------------------------------------------------------------------------
// Simulation parameters
// ---------------------------------------------------------------------------

struct SimConfig {
    num_nodes: usize,
    peers_per_node: usize,
    max_offset_ms: f64,
    max_drift_ppb: i64,
    base_latency_ms: f64,
    latency_spread_ms: f64,
    jitter_ms: f64,
    num_epochs: usize,
    exchanges_per_epoch: usize,
    solver_iterations: u32,
    solver_tolerance_nanos: u64,
}

impl Default for SimConfig {
    fn default() -> Self {
        Self {
            num_nodes: 100,
            peers_per_node: 12,
            max_offset_ms: 500.0,
            max_drift_ppb: 200,
            base_latency_ms: 20.0,
            latency_spread_ms: 30.0,
            jitter_ms: 2.0,
            num_epochs: 20,
            exchanges_per_epoch: 5,
            solver_iterations: 500,
            solver_tolerance_nanos: 100_000, // 100μs
        }
    }
}

// ---------------------------------------------------------------------------
// Build the simulated world
// ---------------------------------------------------------------------------

fn build_network(cfg: &SimConfig, rng: &mut impl Rng) -> SimNetwork {
    let mut nodes = Vec::with_capacity(cfg.num_nodes);

    for i in 0..cfg.num_nodes {
        let mut id_bytes = [0u8; 32];
        id_bytes[0] = (i & 0xFF) as u8;
        id_bytes[1] = ((i >> 8) & 0xFF) as u8;
        rand::fill(&mut id_bytes[2..]);

        let offset_nanos =
            (rng.random_range(-cfg.max_offset_ms..cfg.max_offset_ms)
                * 1_000_000.0) as i64;
        let drift_ppb =
            rng.random_range(-cfg.max_drift_ppb..=cfg.max_drift_ppb);

        nodes.push(SimNode {
            id: NodeId(id_bytes),
            true_offset_nanos: offset_nanos,
            true_drift_ppb: drift_ppb,
            kalman: ClockKalman::new(),
            samr: SAMRState::clock_default(),
            peers: Vec::new(),
            last_solved_offset_nanos: 0,
        });
    }

    // Synthetic latency matrix
    let mut base_latencies =
        vec![vec![0u64; cfg.num_nodes]; cfg.num_nodes];
    for i in 0..cfg.num_nodes {
        for j in (i + 1)..cfg.num_nodes {
            let lat = (cfg.base_latency_ms
                + rng.random_range(0.0..cfg.latency_spread_ms))
                * 1_000_000.0;
            let lat_nanos = lat.max(1_000_000.0) as u64;
            base_latencies[i][j] = lat_nanos;
            base_latencies[j][i] = lat_nanos;
        }
    }

    // Build peer graph: k-nearest by latency + random long links
    let k = cfg.peers_per_node;
    for i in 0..cfg.num_nodes {
        let mut candidates: Vec<(usize, u64)> = (0..cfg.num_nodes)
            .filter(|&j| j != i)
            .map(|j| (j, base_latencies[i][j]))
            .collect();
        candidates.sort_by_key(|&(_, lat)| lat);

        let near = k * 3 / 4;
        let far = k - near;

        let mut peers: Vec<usize> =
            candidates.iter().take(near).map(|&(j, _)| j).collect();

        let remaining: Vec<usize> = candidates
            .iter()
            .skip(near)
            .map(|&(j, _)| j)
            .collect();
        for _ in 0..far.min(remaining.len()) {
            let idx = rng.random_range(0..remaining.len());
            if !peers.contains(&remaining[idx]) {
                peers.push(remaining[idx]);
            }
        }

        nodes[i].peers = peers;
    }

    // Make edges bidirectional
    let mut edges_to_add: Vec<(usize, usize)> = Vec::new();
    for i in 0..cfg.num_nodes {
        for &j in &nodes[i].peers {
            if !nodes[j].peers.contains(&i) {
                edges_to_add.push((j, i));
            }
        }
    }
    for (j, i) in edges_to_add {
        nodes[j].peers.push(i);
    }

    SimNetwork {
        nodes,
        base_latencies,
        jitter_stddev_nanos: cfg.jitter_ms * 1_000_000.0,
    }
}

// ---------------------------------------------------------------------------
// Statistics helpers
// ---------------------------------------------------------------------------

struct EpochStats {
    epoch: usize,
    max_error_us: f64,
    mean_error_us: f64,
    median_error_us: f64,
    p95_error_us: f64,
    p99_error_us: f64,
    solver_max_defect_us: f64,
    solver_converged: bool,
    solver_iterations: u32,
    outliers_detected: usize,
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

// ---------------------------------------------------------------------------
// Main simulation
// ---------------------------------------------------------------------------

fn main() {
    let cfg = SimConfig::default();
    let mut rng = rand::rng();

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  RTFSN Distributed Clock Simulation                    ║");
    println!("╠══════════════════════════════════════════════════════════╣");
    println!("║  Nodes:            {:>6}                               ║", cfg.num_nodes);
    println!("║  Peers/node:       {:>6}                               ║", cfg.peers_per_node);
    println!("║  Max offset:    {:>+8.1} ms                            ║", cfg.max_offset_ms);
    println!("║  Max drift:     {:>+8} ppb                            ║", cfg.max_drift_ppb);
    println!("║  Base latency:  {:>8.1} ms                            ║", cfg.base_latency_ms);
    println!("║  Jitter stddev: {:>8.1} ms                            ║", cfg.jitter_ms);
    println!("║  Epochs:            {:>6}                               ║", cfg.num_epochs);
    println!("║  Exchanges/epoch:   {:>6}                               ║", cfg.exchanges_per_epoch);
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();

    let mut net = build_network(&cfg, &mut rng);

    // Print initial clock distribution
    let mut initial_offsets: Vec<f64> = net
        .nodes
        .iter()
        .map(|n| n.true_offset_nanos as f64 / 1_000_000.0)
        .collect();
    initial_offsets.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let initial_spread =
        initial_offsets.last().unwrap() - initial_offsets.first().unwrap();
    println!(
        "Initial clock spread: {:.1} ms  (min={:.1} ms, max={:.1} ms)",
        initial_spread,
        initial_offsets.first().unwrap(),
        initial_offsets.last().unwrap(),
    );

    // Print peer connectivity stats
    let peer_counts: Vec<usize> =
        net.nodes.iter().map(|n| n.peers.len()).collect();
    let min_peers = *peer_counts.iter().min().unwrap();
    let max_peers = *peer_counts.iter().max().unwrap();
    let avg_peers =
        peer_counts.iter().sum::<usize>() as f64 / peer_counts.len() as f64;
    println!(
        "Peer connectivity: min={}, avg={:.1}, max={}",
        min_peers, avg_peers, max_peers,
    );
    println!();

    let epoch_duration_nanos: i64 = 30_000_000_000; // 30s
    let mut clock_stream = ClockStream::new();
    let mut epoch_boundary =
        EpochBoundarySolver::new(epoch_duration_nanos as u64);
    let mut all_stats: Vec<EpochStats> = Vec::new();

    println!(
        "{:>5} {:>12} {:>12} {:>12} {:>12} {:>12} {:>10} {:>5} {:>4}",
        "Epoch", "Max(μs)", "Mean(μs)", "Med(μs)", "P95(μs)", "P99(μs)",
        "Defect(μs)", "Iter", "Out",
    );
    println!("{}", "-".repeat(100));

    for epoch in 0..cfg.num_epochs {
        let real_time = epoch as i64 * epoch_duration_nanos;

        // ── Phase 1: Measure ──────────────────────────────────────
        // Each node does pairwise exchanges with peers.
        let mut all_twists: Vec<(usize, usize, TwistIndex)> = Vec::new();

        for i in 0..cfg.num_nodes {
            let peers = net.nodes[i].peers.clone();
            for &j in &peers {
                for _ in 0..cfg.exchanges_per_epoch {
                    let twist =
                        net.simulate_exchange(&mut rng, i, j, real_time);
                    all_twists.push((i, j, twist));
                }
            }
        }

        // ── Phase 2: Solve (holonomy) ─────────────────────────────
        // Pick the best (lowest-RTT) twist per edge
        let mut best_twists: HashMap<(usize, usize), TwistIndex> =
            HashMap::new();
        for &(i, j, ref twist) in &all_twists {
            let key = if i < j { (i, j) } else { (j, i) };
            let is_better = best_twists
                .get(&key)
                .map_or(true, |prev| twist.rtt_nanos < prev.rtt_nanos);
            if is_better {
                best_twists.insert(key, twist.clone());
            }
        }

        // Seed solver from previous epoch's output (or zero on epoch 0).
        let mut solver = HolonomySolver::new();
        for node in net.nodes.iter() {
            let seed = node.last_solved_offset_nanos;
            let drift_est = (node.kalman.drift_rate() * 1e9) as i64;
            solver.add_node(
                node.id,
                ChiralFrame::new(seed, drift_est, 100),
            );
        }

        for (&(i, j), twist) in &best_twists {
            solver.add_measurement(
                net.nodes[i].id,
                net.nodes[j].id,
                twist.clone(),
            );
        }

        let solve_result =
            solver.solve(cfg.solver_iterations, cfg.solver_tolerance_nanos);

        // Store solved offsets back into nodes & feed into Kalman
        let dt =
            if epoch == 0 { 1.0 } else { epoch_duration_nanos as f64 / 1e9 };
        for node in &mut net.nodes {
            if let Some(frame) = solver.nodes.get(&node.id) {
                node.last_solved_offset_nanos = frame.offset_nanos;
                let solved_secs = frame.offset_nanos as f64 / 1e9;
                let measurement_var = (solve_result.mean_defect_nanos as f64
                    / 1e9)
                    .powi(2)
                    .max(1e-12);
                node.kalman.step(dt, solved_secs, measurement_var);
            }
        }

        // SAMR classification
        for idx in 0..net.nodes.len() {
            let offset_nanos = net.nodes[idx].last_solved_offset_nanos;
            let kalman_unc = net.nodes[idx].kalman.uncertainty();
            let kalman_drift = net.nodes[idx].kalman.drift_rate();
            let uncertainty_nanos = (kalman_unc * 1e9) as u64;
            let drift_ppb = (kalman_drift * 1e9) as i64;

            net.nodes[idx].samr.classify_reachability(true);
            if !net.nodes[idx].peers.is_empty() {
                let first_peer = net.nodes[idx].peers[0];
                net.nodes[idx]
                    .samr
                    .classify_tier(net.base_latencies[idx][first_peer]);
            }
            net.nodes[idx]
                .samr
                .classify_offset(offset_nanos, uncertainty_nanos);
            net.nodes[idx].samr.classify_drift(drift_ppb);
        }

        // Outlier detection: scale-aware threshold
        let defective = solver.detect_defective_nodes(
            solve_result
                .mean_defect_nanos
                .saturating_mul(5)
                .max(cfg.solver_tolerance_nanos * 10),
        );

        // Marzullo intersection for consensus interval
        let intervals: Vec<TimeInterval> = solver
            .nodes
            .values()
            .map(|frame| {
                let offset_secs = frame.offset_secs();
                let unc = solve_result.mean_defect_nanos as f64 / 1e9;
                TimeInterval::from_measurement(offset_secs, unc.max(0.001))
            })
            .collect();
        let _marzullo_result = marzullo(&intervals);

        // ── Phase 3: Emit clock tick ──────────────────────────────
        let node_offsets: Vec<i64> =
            solver.nodes.values().map(|f| f.offset_nanos).collect();
        let node_drifts: Vec<i64> =
            solver.nodes.values().map(|f| f.drift_ppb).collect();

        let mirror = epoch_boundary.solve_boundary(
            &node_offsets,
            &node_drifts,
            real_time as u64,
        );
        let epoch_hash = {
            let mut h = blake3::Hasher::new();
            for offset in &node_offsets {
                h.update(&offset.to_le_bytes());
            }
            *h.finalize().as_bytes()
        };
        clock_stream.emit(
            mirror.fixed_point_nanos,
            mirror.disagreement_nanos,
            epoch_hash,
            None,
        );

        // ── Error computation ─────────────────────────────────────
        // The solver determines relative offsets, not absolute (one
        // global constant is underdetermined). Center both distributions
        // and compare — this measures how well the network resolves
        // the relative clock relationships.
        let pairs: Vec<(i64, i64)> = net
            .nodes
            .iter()
            .filter_map(|node| {
                solver
                    .nodes
                    .get(&node.id)
                    .map(|frame| (node.true_offset_nanos, frame.offset_nanos))
            })
            .collect();
        let n = pairs.len() as f64;
        let true_mean =
            pairs.iter().map(|(t, _)| *t as f64).sum::<f64>() / n;
        let solved_mean =
            pairs.iter().map(|(_, s)| *s as f64).sum::<f64>() / n;

        let mut errors_us: Vec<f64> = pairs
            .iter()
            .map(|(t, s)| {
                let true_centered = *t as f64 - true_mean;
                let solved_centered = *s as f64 - solved_mean;
                (true_centered - solved_centered).abs() / 1_000.0
            })
            .collect();
        errors_us.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let stats = EpochStats {
            epoch,
            max_error_us: errors_us.last().copied().unwrap_or(0.0),
            mean_error_us: errors_us.iter().sum::<f64>()
                / errors_us.len().max(1) as f64,
            median_error_us: percentile(&errors_us, 0.5),
            p95_error_us: percentile(&errors_us, 0.95),
            p99_error_us: percentile(&errors_us, 0.99),
            solver_max_defect_us: solve_result.max_defect_nanos as f64
                / 1_000.0,
            solver_converged: solve_result.converged,
            solver_iterations: solve_result.iterations,
            outliers_detected: defective.len(),
        };

        println!(
            "{:>5} {:>12.1} {:>12.1} {:>12.1} {:>12.1} {:>12.1} {:>10.1} {:>5} {:>4}",
            stats.epoch,
            stats.max_error_us,
            stats.mean_error_us,
            stats.median_error_us,
            stats.p95_error_us,
            stats.p99_error_us,
            stats.solver_max_defect_us,
            stats.solver_iterations,
            stats.outliers_detected,
        );

        all_stats.push(stats);
    }

    // ── Final report ──────────────────────────────────────────────
    println!();
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  Results                                               ║");
    println!("╠══════════════════════════════════════════════════════════╣");

    // Skip epoch 0 (no measurements yet — numbers are meaningless)
    let first_real = if all_stats.len() > 1 {
        &all_stats[1]
    } else {
        &all_stats[0]
    };
    let last = all_stats.last().unwrap();

    let initial_spread_us = initial_spread * 1_000.0; // ms → μs

    println!(
        "║  Initial spread:     {:>10.0} μs ({:.1} ms)            ║",
        initial_spread_us, initial_spread,
    );
    println!(
        "║  Epoch {:>2} → max: {:>10.1} μs  mean: {:>10.1} μs      ║",
        first_real.epoch,
        first_real.max_error_us,
        first_real.mean_error_us,
    );
    println!(
        "║  Epoch {:>2} → max: {:>10.1} μs  mean: {:>10.1} μs      ║",
        last.epoch, last.max_error_us, last.mean_error_us,
    );

    let reduction = if initial_spread_us > 0.0 {
        (1.0 - last.mean_error_us / initial_spread_us) * 100.0
    } else {
        0.0
    };
    println!(
        "║  Reduction vs spread: {:>6.2}%                          ║",
        reduction,
    );

    let converged_count =
        all_stats.iter().filter(|s| s.solver_converged).count();
    println!(
        "║  Solver converged:     {}/{} epochs                    ║",
        converged_count, cfg.num_epochs,
    );

    let chain_valid = clock_stream.verify_all();
    println!(
        "║  Clock chain:          {} ticks, valid={}              ║",
        clock_stream.len(),
        if chain_valid { "yes" } else { " no" },
    );

    let samr_codes: Vec<u64> =
        net.nodes.iter().map(|n| n.samr.crt_decode()).collect();
    let unique_samr = {
        let mut s = samr_codes.clone();
        s.sort();
        s.dedup();
        s.len()
    };
    println!(
        "║  SAMR states:          {} unique / {} possible          ║",
        unique_samr,
        SAMRState::clock_default().total_resolution(),
    );
    println!("╚══════════════════════════════════════════════════════════╝");

    // Best epoch
    let best = all_stats
        .iter()
        .min_by(|a, b| {
            a.mean_error_us.partial_cmp(&b.mean_error_us).unwrap()
        })
        .unwrap();
    println!();
    println!(
        "Best epoch: {} — mean {:.1} μs, max {:.1} μs, median {:.1} μs",
        best.epoch, best.mean_error_us, best.max_error_us, best.median_error_us,
    );

    // Top 10 worst-converged nodes at final epoch
    println!();
    println!("Top 10 worst nodes at final epoch:");
    println!(
        "{:>6} {:>12} {:>12} {:>12} {:>10} {:>6}",
        "Node", "TrueOff(ms)", "Solved(ms)", "Error(μs)", "Drift(ppb)", "Peers",
    );
    println!("{}", "-".repeat(70));

    // Center for final report
    let final_true_mean = net
        .nodes
        .iter()
        .map(|n| n.true_offset_nanos as f64)
        .sum::<f64>()
        / net.nodes.len() as f64;
    let final_solved_mean = net
        .nodes
        .iter()
        .map(|n| n.last_solved_offset_nanos as f64)
        .sum::<f64>()
        / net.nodes.len() as f64;

    let mut node_errors: Vec<(usize, f64)> = net
        .nodes
        .iter()
        .enumerate()
        .map(|(i, node)| {
            let tc = node.true_offset_nanos as f64 - final_true_mean;
            let sc =
                node.last_solved_offset_nanos as f64 - final_solved_mean;
            let err_us = (tc - sc).abs() / 1_000.0;
            (i, err_us)
        })
        .collect();
    node_errors.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    for &(i, err_us) in node_errors.iter().take(10) {
        let node = &net.nodes[i];
        println!(
            "{:>6} {:>12.3} {:>12.3} {:>12.1} {:>10} {:>6}",
            i,
            node.true_offset_nanos as f64 / 1e6,
            node.last_solved_offset_nanos as f64 / 1e6,
            err_us,
            node.true_drift_ppb,
            node.peers.len(),
        );
    }

    // Top 10 best-converged
    println!();
    println!("Top 10 best nodes at final epoch:");
    println!(
        "{:>6} {:>12} {:>12} {:>12} {:>10} {:>6}",
        "Node", "TrueOff(ms)", "Solved(ms)", "Error(μs)", "Drift(ppb)", "Peers",
    );
    println!("{}", "-".repeat(70));

    node_errors.reverse();
    for &(i, err_us) in node_errors.iter().take(10) {
        let node = &net.nodes[i];
        println!(
            "{:>6} {:>12.3} {:>12.3} {:>12.1} {:>10} {:>6}",
            i,
            node.true_offset_nanos as f64 / 1e6,
            node.last_solved_offset_nanos as f64 / 1e6,
            err_us,
            node.true_drift_ppb,
            node.peers.len(),
        );
    }
}
