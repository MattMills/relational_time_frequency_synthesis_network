// Geoid circulation simulation: demonstrates the bidirectional
// network topology model and privacy-preserving refinement protocol.
//
// Simulates 30 nodes in 3 geographic regions with known inter-region
// latencies, then shows how the NetworkGeoid hierarchy forms, stabilises
// across epochs, and drives a downward measurement schedule.

use rand::Rng;
use rand_distr::{Distribution, Normal};

use rtfsn_core::crypto::blinding::BlindingSecret;
use rtfsn_core::geoid::circulation::CirculationManager;
use rtfsn_core::holonomy::twist::{TwistIndex, TwistLUT};
use rtfsn_core::layers::layer1::Layer1Beacon;
use rtfsn_core::sync::geometry::CoordinateState;
use rtfsn_core::types::{Coordinates, Epoch, NodeId, COORDINATE_DIMENSIONS};

// ─────────────────────────────────────────────────────────────────────────────
// Simulation parameters
// ─────────────────────────────────────────────────────────────────────────────

const NODES_PER_REGION: usize = 10;
const NUM_EPOCHS: usize = 8;
const GEOID_LEVELS: u8 = 4;

// Coordinate centroids (seconds ≈ latency)
// A "West":    [0.001, 0.000] – intra ~1ms
// B "Central": [0.010, 0.000] – A↔B ~9ms  (merges with A at L2, 32ms threshold)
// C "East":    [0.050, 0.010] – A↔C ~52ms, B↔C ~40ms (stays separate until L3)
const REGION_NAMES: [&str; 3] = ["West    ", "Central ", "East    "];
const REGION_CENTROIDS: [[f64; 2]; 3] = [
    [0.001, 0.000],
    [0.010, 0.000],
    [0.050, 0.010],
];
const INTRA_JITTER: f64 = 0.0003; // ±0.3ms coordinate jitter per node

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

fn make_node_id(region: u8, idx: u8) -> NodeId {
    let mut bytes = [0u8; 32];
    bytes[0] = region;
    bytes[1] = idx;
    NodeId(bytes)
}

fn make_coords(center: [f64; 2], jitter: f64, rng: &mut impl Rng) -> Coordinates {
    let dist = Normal::new(0.0, jitter).unwrap();
    let mut dims = [0.0f64; COORDINATE_DIMENSIONS];
    dims[0] = center[0] + dist.sample(rng);
    dims[1] = center[1] + dist.sample(rng);
    for d in &mut dims[2..] {
        *d = dist.sample(rng) * 0.1;
    }
    Coordinates { dims }
}

fn make_beacon(
    node_id: &NodeId,
    epoch: Epoch,
    position: Coordinates,
    clock_offset: f64,
) -> Layer1Beacon {
    let secret = BlindingSecret::new(node_id, epoch);
    let blinded_id = secret.blind();
    let proof = secret.prove_knowledge();
    Layer1Beacon {
        blinded_id,
        epoch,
        coordinates: CoordinateState {
            position,
            error: 0.01,
        },
        clock_offset,
        clock_drift: 0.0,
        uncertainty: 0.001,
        peer_count: 5,
        consistency_score: 0.95,
        proof_of_identity: proof,
        proof_of_peer_count: None,
        geoid_coord: None,
    }
}

fn centroid(region: usize) -> Coordinates {
    let mut dims = [0.0f64; COORDINATE_DIMENSIONS];
    dims[0] = REGION_CENTROIDS[region][0];
    dims[1] = REGION_CENTROIDS[region][1];
    Coordinates { dims }
}

fn dist_ms(a: &Coordinates, b: &Coordinates) -> f64 {
    a.distance(b) * 1000.0
}

fn bytes_to_hex(bytes: &[u8]) -> String {
    bytes
        .iter()
        .map(|b| format!("{:02x}", b))
        .collect::<String>()
}

// ─────────────────────────────────────────────────────────────────────────────
// Structural stability helpers
// ─────────────────────────────────────────────────────────────────────────────

// A snapshot of geoid topology independent of epoch-specific IDs.
// Region IDs embed the epoch for privacy (unlinkable across epochs), so we
// compare structure by centroid proximity rather than ID equality.
#[derive(Clone)]
struct GeoidSnapshot {
    // Per level: (centroid_x_ms, centroid_y_ms, member_count)
    levels: Vec<Vec<(f64, f64, u32)>>,
}

impl GeoidSnapshot {
    fn from_geoid(g: &rtfsn_core::geoid::model::NetworkGeoid) -> Self {
        Self {
            levels: g
                .layers
                .iter()
                .map(|l| {
                    l.regions
                        .iter()
                        .map(|r| {
                            (
                                r.centroid.dims[0] * 1000.0,
                                r.centroid.dims[1] * 1000.0,
                                r.member_count,
                            )
                        })
                        .collect()
                })
                .collect(),
        }
    }

    // Normalised structural delta [0.0, 1.0]:
    // 0 = identical structure, 1 = completely different.
    fn delta(&self, prev: &GeoidSnapshot) -> f64 {
        if self.levels.len() != prev.levels.len() {
            return 1.0;
        }
        let mut total_move = 0.0_f64;
        let mut total_count = 0_usize;
        for (cur_level, prev_level) in self.levels.iter().zip(prev.levels.iter()) {
            if cur_level.len() != prev_level.len() {
                // Region count changed — maximum change at this level
                total_move += cur_level.len().max(prev_level.len()) as f64;
                total_count += cur_level.len().max(prev_level.len());
                continue;
            }
            // Match each current region to the closest previous region
            for &(cx, cy, _) in cur_level {
                let nearest_move = prev_level
                    .iter()
                    .map(|&(px, py, _)| ((cx - px).powi(2) + (cy - py).powi(2)).sqrt())
                    .fold(f64::MAX, f64::min);
                total_move += nearest_move;
                total_count += 1;
            }
        }
        if total_count == 0 {
            return 0.0;
        }
        // Normalise: 1ms of centroid movement → ~0.05 delta
        (total_move / total_count as f64 / 20.0).min(1.0)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────

fn main() {
    let mut rng = rand::rng();

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  RTFSN Geoid Circulation Simulation                    ║");
    println!("╠══════════════════════════════════════════════════════════╣");
    println!(
        "║  {} nodes × 3 regions = {}  |  {} levels  |  {} epochs  ║",
        NODES_PER_REGION,
        NODES_PER_REGION * 3,
        GEOID_LEVELS,
        NUM_EPOCHS
    );
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();

    // ── Region topology ───────────────────────────────────────────
    println!("Network topology (Vivaldi coordinate space; distance ≈ latency):");
    let ca = centroid(0);
    let cb = centroid(1);
    let cc = centroid(2);

    println!(
        "  A \"{name}\" {n} nodes  centroid [{cx:.3},{cy:.3}]ms  intra~{intra:.1}ms",
        name = REGION_NAMES[0],
        n = NODES_PER_REGION,
        cx = REGION_CENTROIDS[0][0] * 1000.0,
        cy = REGION_CENTROIDS[0][1] * 1000.0,
        intra = INTRA_JITTER * 1000.0 * 3.0,
    );
    println!(
        "  B \"{name}\" {n} nodes  centroid [{cx:.3},{cy:.3}]ms  intra~{intra:.1}ms  A↔B~{ab:.1}ms",
        name = REGION_NAMES[1],
        n = NODES_PER_REGION,
        cx = REGION_CENTROIDS[1][0] * 1000.0,
        cy = REGION_CENTROIDS[1][1] * 1000.0,
        intra = INTRA_JITTER * 1000.0 * 3.0,
        ab = dist_ms(&ca, &cb),
    );
    println!(
        "  C \"{name}\" {n} nodes  centroid [{cx:.3},{cy:.3}]ms  intra~{intra:.1}ms  A↔C~{ac:.1}ms  B↔C~{bc:.1}ms",
        name = REGION_NAMES[2],
        n = NODES_PER_REGION,
        cx = REGION_CENTROIDS[2][0] * 1000.0,
        cy = REGION_CENTROIDS[2][1] * 1000.0,
        intra = INTRA_JITTER * 1000.0 * 3.0,
        ac = dist_ms(&ca, &cc),
        bc = dist_ms(&cb, &cc),
    );
    println!();
    println!("Expected hierarchy (CoarseningOperator defaults: 2ms base, 4× per level):");
    println!("  L0 (2ms):   3 regions   — each geographic cluster forms one fine region");
    println!("  L1 (8ms):   3 regions   — A-B distance 9ms > 8ms threshold, stay separate");
    println!("  L2 (32ms):  2 regions   — A+B merge (9ms < 32ms), C remains separate");
    println!("  L3 (128ms): 1 region    — all merge (A-C 52ms < 128ms)");
    println!();

    // ── Generate stable node positions (same across epochs) ───────
    let node_positions: Vec<(u8, Coordinates)> = (0..3)
        .flat_map(|r| {
            (0..NODES_PER_REGION)
                .map(move |_| r as u8)
                .collect::<Vec<_>>()
        })
        .map(|r| {
            let pos = make_coords(REGION_CENTROIDS[r as usize], INTRA_JITTER, &mut rng);
            (r, pos)
        })
        .collect();

    // ── Epoch loop ────────────────────────────────────────────────
    let mut circulation = CirculationManager::with_levels(Epoch(0), GEOID_LEVELS);
    let mut prev_snapshot = GeoidSnapshot::from_geoid(&circulation.geoid);

    println!(
        "{:>6}  {:>4} {:>4} {:>4} {:>4}  {:>7}  {:>5}  {:>8}  {:>9}",
        "Epoch", "L0", "L1", "L2", "L3", "Delta", "Hints", "Schedule", "Status"
    );
    println!("{}", "─".repeat(72));

    for epoch_num in 1..=NUM_EPOCHS {
        let epoch = Epoch(epoch_num as u64);

        // Build L1 beacons (each node's Vivaldi position, blinded)
        let beacons: Vec<Layer1Beacon> = node_positions
            .iter()
            .enumerate()
            .map(|(i, (region_idx, pos))| {
                let node_id = make_node_id(*region_idx, i as u8);
                let clock_offset = pos.dims[0] * 0.001;
                make_beacon(&node_id, epoch, pos.clone(), clock_offset)
            })
            .collect();
        let beacon_refs: Vec<&Layer1Beacon> = beacons.iter().collect();

        // Upward pass: build the geoid hierarchy
        circulation.process_upward(&beacon_refs, epoch);

        // Structural delta: compare centroid positions (IDs change each epoch by design
        // for privacy, so we compare topology rather than IDs).
        let snap = GeoidSnapshot::from_geoid(&circulation.geoid);
        let delta = snap.delta(&prev_snapshot);
        prev_snapshot = snap;

        // Region counts per level
        let level_counts: Vec<usize> = (0..GEOID_LEVELS as usize)
            .map(|l| {
                circulation
                    .geoid
                    .layers
                    .get(l)
                    .map_or(0, |layer| layer.regions.len())
            })
            .collect();

        // Downward pass for node A[0]
        let a0_pos = &node_positions[0].1;
        let hints = circulation.generate_hints(a0_pos);
        let schedule = circulation.apply_hints(&hints);

        let status = if delta < 0.01 { "converged" } else { "" };

        println!(
            "{:>6}  {:>4} {:>4} {:>4} {:>4}  {:>7.3}  {:>5}  {:>5} meas  {:>9}",
            epoch_num,
            level_counts[0],
            level_counts[1],
            level_counts[2],
            level_counts[3],
            delta,
            hints.len(),
            schedule.total_measurements(),
            status,
        );
    }

    // ── Geoid snapshot ────────────────────────────────────────────
    println!();
    println!("Geoid snapshot (epoch {}):", NUM_EPOCHS);
    for layer in &circulation.geoid.layers {
        let res_ms = layer.resolution_nanos as f64 / 1_000_000.0;
        println!(
            "  Level {} ({:.0}ms resolution): {} region(s)",
            layer.level,
            res_ms,
            layer.regions.len()
        );
        for region in &layer.regions {
            let cx = region.centroid.dims[0] * 1000.0;
            let cy = region.centroid.dims[1] * 1000.0;

            let edge_str = if region.inter_edges.is_empty() {
                "(no neighbors)".to_string()
            } else {
                region
                    .inter_edges
                    .iter()
                    .map(|e| format!("{:.1}ms", e.mean_nanos_plaintext as f64 / 1_000_000.0))
                    .collect::<Vec<_>>()
                    .join(", ")
            };

            println!(
                "    members={:>3}  centroid=({:>7.3},{:>6.3})ms  edges→[{}]",
                region.member_count, cx, cy, edge_str,
            );
        }
    }

    // ── Accuracy check: geoid edge distances vs known latencies ───
    // L0 edges only span regions within 3× L0-threshold (6ms); A-B at ~9ms is
    // above this, so inter-region edges first appear at L1 (3×8ms = 24ms threshold).
    println!();
    println!("Geoid accuracy: inter-region edges (L0 edge threshold 6ms < 9ms A-B distance,");
    println!("                so cross-region edges first appear at L1, threshold 24ms)");
    println!(
        "  True distances:  A-B {:.1}ms   A-C {:.1}ms   B-C {:.1}ms",
        dist_ms(&ca, &cb),
        dist_ms(&ca, &cc),
        dist_ms(&cb, &cc),
    );
    // Show edges at each level where they appear
    for layer in &circulation.geoid.layers {
        let has_cross = layer.regions.iter().any(|r| !r.inter_edges.is_empty());
        if !has_cross {
            continue;
        }
        let res_ms = layer.resolution_nanos as f64 / 1_000_000.0;
        println!("  L{} ({:.0}ms) edges:", layer.level, res_ms);
        let mut shown: std::collections::HashSet<(usize, usize)> =
            std::collections::HashSet::new();
        for (i, region) in layer.regions.iter().enumerate() {
            for edge in &region.inter_edges {
                if let Some(j) = layer.regions.iter().position(|r| r.id == edge.to) {
                    let pair = if i < j { (i, j) } else { (j, i) };
                    if shown.insert(pair) {
                        let edge_ms = edge.mean_nanos_plaintext as f64 / 1_000_000.0;
                        let fcx = region.centroid.dims[0] * 1000.0;
                        let tcx = layer.regions[j].centroid.dims[0] * 1000.0;
                        println!(
                            "    [{:>7.2}ms] ↔ [{:>7.2}ms]:  geoid distance {:.2}ms",
                            fcx, tcx, edge_ms,
                        );
                    }
                }
            }
        }
    }

    // ── Refinement hints detail ───────────────────────────────────
    println!();
    println!("Refinement hints for node A[0] (West region):");
    let a0_pos = &node_positions[0].1;
    let hints = circulation.generate_hints(a0_pos);
    if hints.is_empty() {
        println!("  (none — geoid fully converged for this node)");
    } else {
        for h in &hints {
            println!(
                "  Level {:>1}: region residual {:>8.3}ms  priority {}  {} neighbor(s) suggested",
                h.geoid_level,
                h.geoid_residual_nanos as f64 / 1_000_000.0,
                h.priority,
                h.suggested_neighbor_regions.len(),
            );
        }
    }
    let schedule = circulation.apply_hints(&hints);
    println!(
        "  → Measurement schedule: {} base + {} guided = {} total/epoch",
        schedule.min_measurements,
        schedule.total_measurements() - schedule.min_measurements,
        schedule.total_measurements(),
    );

    // Show how guided schedule differs for a node near a region boundary
    let boundary_pos = {
        // Node sitting halfway between A and B (high boundary uncertainty)
        let mut dims = [0.0f64; COORDINATE_DIMENSIONS];
        dims[0] = (REGION_CENTROIDS[0][0] + REGION_CENTROIDS[1][0]) / 2.0;
        dims[1] = 0.0;
        Coordinates { dims }
    };
    let boundary_hints = circulation.generate_hints(&boundary_pos);
    let boundary_schedule = circulation.apply_hints(&boundary_hints);
    println!();
    println!(
        "  Boundary node (A-B midpoint {:.2}ms):  {} hints → {} meas/epoch",
        boundary_pos.dims[0] * 1000.0,
        boundary_hints.len(),
        boundary_schedule.total_measurements(),
    );

    // ── RelationalLatencyProfile ──────────────────────────────────
    println!();
    println!(
        "RelationalLatencyProfile: {} NTP exchanges per pair per epoch over {} epochs",
        5, NUM_EPOCHS
    );

    let node_a0 = make_node_id(0, 0);
    let node_b0 = make_node_id(1, 0);
    let node_c0 = make_node_id(2, 0);

    let a0 = &node_positions[0].1;
    let b0 = &node_positions[NODES_PER_REGION].1;
    let c0 = &node_positions[NODES_PER_REGION * 2].1;

    // True one-way latencies (seconds → nanoseconds)
    let rtt_ab_true = a0.distance(b0) * 2.0 * 1e9;
    let rtt_ac_true = a0.distance(c0) * 2.0 * 1e9;

    let jitter = Normal::new(0.0_f64, 50_000.0).unwrap(); // 50 μs std dev

    let mut lut = TwistLUT::new();
    for ep in 1..=(NUM_EPOCHS as u64) {
        // Simulate a slight latency increase on A↔C (trend test)
        let ab_drift = 0.0_f64;
        let ac_drift = (ep - 1) as f64 * 5_000.0; // +5μs per epoch

        for _ in 0..5 {
            lut.insert(
                node_a0,
                node_b0,
                TwistIndex {
                    rtt_nanos: (rtt_ab_true + ab_drift + jitter.sample(&mut rng).abs()) as u64,
                    offset_nanos: (jitter.sample(&mut rng) * 0.05) as i64,
                    asymmetry_nanos: (jitter.sample(&mut rng) * 0.02) as i64,
                    quality: 95,
                    epoch_measured: ep,
                },
            );
            lut.insert(
                node_a0,
                node_c0,
                TwistIndex {
                    rtt_nanos: (rtt_ac_true + ac_drift + jitter.sample(&mut rng).abs()) as u64,
                    offset_nanos: (jitter.sample(&mut rng) * 0.05) as i64,
                    asymmetry_nanos: (jitter.sample(&mut rng) * 0.02) as i64,
                    quality: 88,
                    epoch_measured: ep,
                },
            );
        }
    }

    println!(
        "{:>8}  {:>10}  {:>10}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}",
        "Pair",
        "True RTT",
        "Mean RTT",
        "StdDev",
        "P10",
        "P50",
        "P90",
        "Trend"
    );
    println!("{}", "─".repeat(86));

    if let Some(p) = lut.profile(node_a0, node_b0) {
        let stddev_ms = (p.variance_nanos as f64).sqrt() / 1_000_000.0;
        println!(
            "{:>8}  {:>10.3}  {:>10.3}  {:>8.4}  {:>8.3}  {:>8.3}  {:>8.3}  {:>+7.3}",
            "A↔B",
            rtt_ab_true / 1_000_000.0,
            p.mean_nanos as f64 / 1_000_000.0,
            stddev_ms,
            p.p10_nanos as f64 / 1_000_000.0,
            p.p50_nanos as f64 / 1_000_000.0,
            p.p90_nanos as f64 / 1_000_000.0,
            p.trend_nanos_per_epoch as f64 / 1_000_000.0,
        );
    }
    if let Some(p) = lut.profile(node_a0, node_c0) {
        let stddev_ms = (p.variance_nanos as f64).sqrt() / 1_000_000.0;
        println!(
            "{:>8}  {:>10.3}  {:>10.3}  {:>8.4}  {:>8.3}  {:>8.3}  {:>8.3}  {:>+7.3}",
            "A↔C",
            rtt_ac_true / 1_000_000.0,
            p.mean_nanos as f64 / 1_000_000.0,
            stddev_ms,
            p.p10_nanos as f64 / 1_000_000.0,
            p.p50_nanos as f64 / 1_000_000.0,
            p.p90_nanos as f64 / 1_000_000.0,
            p.trend_nanos_per_epoch as f64 / 1_000_000.0,
        );
    }
    println!("  (RTT values in ms; trend in ms/epoch; A↔C has +5μs/epoch injected drift)");

    // ── Summary ───────────────────────────────────────────────────
    let total_regions: usize = circulation
        .geoid
        .layers
        .iter()
        .map(|l| l.regions.len())
        .sum();
    let hash_hex = bytes_to_hex(&circulation.geoid.hash()[..8]);

    println!();
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  Results                                               ║");
    println!("╠══════════════════════════════════════════════════════════╣");
    println!(
        "║  Geoid depth:          {} levels                        ║",
        circulation.geoid.depth()
    );
    println!(
        "║  Total regions:        {} across all levels             ║",
        total_regions
    );
    println!(
        "║  Refinement generation {}                               ║",
        circulation.geoid.refinement_generation
    );
    println!(
        "║  Geoid hash (8 bytes): {}                     ║",
        hash_hex
    );
    let final_snap = GeoidSnapshot::from_geoid(&circulation.geoid);
    let final_delta = final_snap.delta(&prev_snapshot);
    println!(
        "║  Convergence delta:    {:.3}  {}                         ║",
        final_delta,
        if final_delta < 0.01 { "(stable)" } else { "(drifting)" },
    );
    println!("╚══════════════════════════════════════════════════════════╝");
}
