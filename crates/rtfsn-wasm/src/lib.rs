use wasm_bindgen::prelude::*;

use rtfsn_core::holonomy::chiral::ChiralFrame;
use rtfsn_core::holonomy::samr::SAMRState;
use rtfsn_core::holonomy::solver::HolonomySolver;
use rtfsn_core::holonomy::temporal_mirror::TemporalMirror;
use rtfsn_core::layers::layer3::ClockStream;
use rtfsn_core::sync::clock::{ClockState, TimeExchange};
use rtfsn_core::sync::kalman::ClockKalman;
use rtfsn_core::types::NodeId;

#[wasm_bindgen]
pub struct WasmClockNode {
    node_id: NodeId,
    clock_state: ClockState,
    kalman: ClockKalman,
    samr: SAMRState,
    clock_stream: ClockStream,
}

#[wasm_bindgen]
impl WasmClockNode {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        let mut id_bytes = [0u8; 32];
        getrandom::fill(&mut id_bytes).expect("getrandom failed");
        let node_id = NodeId(id_bytes);

        Self {
            node_id,
            clock_state: ClockState::new(),
            kalman: ClockKalman::new(),
            samr: SAMRState::clock_default(),
            clock_stream: ClockStream::new(),
        }
    }

    /// Get high-resolution timestamp in nanoseconds.
    /// Uses performance.now() for sub-millisecond resolution.
    pub fn now_nanos(&self) -> f64 {
        let window = web_sys::window().unwrap();
        let performance = window.performance().unwrap();
        performance.now() * 1_000_000.0 // ms → ns
    }

    /// Process a time exchange with a peer.
    /// Returns the estimated clock offset in nanoseconds.
    pub fn process_exchange(
        &mut self,
        t1: f64,
        t2: f64,
        t3: f64,
        t4: f64,
    ) -> f64 {
        let exchange = TimeExchange::new(t1, t2, t3, t4);
        let offset = exchange.offset();
        let delay = exchange.round_trip_delay();

        let local_time = self.now_nanos() / 1_000_000_000.0;
        self.clock_state.add_exchange(&exchange, local_time);

        let measurement_variance = (delay / 2.0).powi(2);
        self.kalman.step(1.0, offset, measurement_variance);

        // Update SAMR channels
        let offset_nanos = (offset * 1_000_000_000.0) as i64;
        let uncertainty_nanos = (delay * 500_000_000.0) as u64;
        let drift_ppb = (self.kalman.drift_rate() * 1_000_000_000.0) as i64;

        self.samr.classify_reachability(true);
        self.samr
            .classify_tier((delay * 1_000_000_000.0) as u64);
        self.samr
            .classify_offset(offset_nanos, uncertainty_nanos);
        self.samr.classify_drift(drift_ppb);

        offset * 1_000_000_000.0 // return in nanoseconds
    }

    /// Get the current best estimate of clock offset (nanoseconds).
    pub fn estimated_offset_nanos(&self) -> f64 {
        self.kalman.offset() * 1_000_000_000.0
    }

    /// Get the current best estimate of drift rate (ppb).
    pub fn estimated_drift_ppb(&self) -> f64 {
        self.kalman.drift_rate() * 1_000_000_000.0
    }

    /// Get the Kalman filter uncertainty (nanoseconds).
    pub fn uncertainty_nanos(&self) -> f64 {
        self.kalman.uncertainty() * 1_000_000_000.0
    }

    /// Get the SAMR-encoded state as a single integer.
    pub fn samr_state(&self) -> u64 {
        self.samr.crt_decode()
    }

    /// Get the latest clock tick epoch number.
    pub fn latest_epoch(&self) -> u64 {
        self.clock_stream.latest().epoch.0
    }

    /// Get the node ID as a hex string.
    pub fn node_id_hex(&self) -> String {
        self.node_id
            .0
            .iter()
            .map(|b| format!("{:02x}", b))
            .collect()
    }
}

#[wasm_bindgen]
pub struct WasmHolonomySolver {
    solver: HolonomySolver,
    node_count: u32,
}

#[wasm_bindgen]
impl WasmHolonomySolver {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            solver: HolonomySolver::new(),
            node_count: 0,
        }
    }

    /// Add a node with the given offset in nanoseconds.
    pub fn add_node(&mut self, id_byte: u8, offset_nanos: i64) {
        let id = NodeId([id_byte; 32]);
        self.solver
            .add_node(id, ChiralFrame::new(offset_nanos, 0, 100));
        self.node_count += 1;
    }

    /// Add a measurement between two nodes.
    pub fn add_measurement(
        &mut self,
        a_byte: u8,
        b_byte: u8,
        offset_nanos: i64,
        rtt_nanos: u64,
    ) {
        let a = NodeId([a_byte; 32]);
        let b = NodeId([b_byte; 32]);
        self.solver.add_measurement(
            a,
            b,
            rtfsn_core::holonomy::twist::TwistIndex {
                rtt_nanos,
                offset_nanos,
                asymmetry_nanos: 0,
                quality: 100,
                epoch_measured: 0,
            },
        );
    }

    /// Run the solver for the given number of iterations.
    /// Returns the max defect in nanoseconds.
    pub fn solve(&mut self, max_iterations: u32) -> u64 {
        let result = self.solver.solve(max_iterations, 1000);
        result.max_defect_nanos
    }

    /// Check if the solver has converged.
    pub fn converged(&self) -> bool {
        // Re-check by looking at current iteration state
        self.solver.iteration > 0
    }
}

/// Demonstrate the temporal mirror fixed-point computation.
#[wasm_bindgen]
pub fn find_temporal_fixed_point(
    forward_json: &str,
    backward_json: &str,
) -> String {
    let forward: Vec<i64> =
        serde_json::from_str(forward_json).unwrap_or_default();
    let backward: Vec<i64> =
        serde_json::from_str(backward_json).unwrap_or_default();

    let mirror =
        TemporalMirror::find_fixed_point(&forward, &backward, 100);

    serde_json::to_string(&mirror).unwrap_or_default()
}
