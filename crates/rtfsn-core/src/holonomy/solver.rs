use std::collections::HashMap;

use crate::holonomy::chiral::ChiralFrame;
use crate::holonomy::twist::TwistLUT;
use crate::types::NodeId;

/// HolonomySolver: solves for globally consistent clock offsets
/// by minimizing holonomy defects across the network graph.
///
/// This is the same solve as the power grid coherence problem:
/// find the set of node states (clock offsets) such that the
/// twist (measurement residual) around every loop is minimal.
pub struct HolonomySolver {
    pub nodes: HashMap<NodeId, ChiralFrame>,
    pub lut: TwistLUT,
    pub iteration: u32,
    damping: f64,
}

#[derive(Debug, Clone)]
pub struct SolveResult {
    pub iterations: u32,
    pub max_defect_nanos: u64,
    pub mean_defect_nanos: u64,
    pub converged: bool,
}

impl HolonomySolver {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            lut: TwistLUT::new(),
            iteration: 0,
            damping: 0.3,
        }
    }

    pub fn add_node(&mut self, id: NodeId, frame: ChiralFrame) {
        self.nodes.insert(id, frame);
    }

    pub fn add_measurement(
        &mut self,
        a: NodeId,
        b: NodeId,
        twist: crate::holonomy::twist::TwistIndex,
    ) {
        self.lut.insert(a, b, twist);
    }

    /// One iteration of the solve: for each node, adjust its offset
    /// to minimize disagreement with all neighbors.
    pub fn step(&mut self) -> SolveResult {
        let node_ids: Vec<NodeId> = self.nodes.keys().copied().collect();
        let mut updates: HashMap<NodeId, i64> = HashMap::new();
        let mut defects: Vec<u64> = Vec::new();

        for &node_id in &node_ids {
            let neighbors = self.lut.neighbors(node_id);
            if neighbors.is_empty() {
                continue;
            }

            let node_frame = &self.nodes[&node_id];
            let mut correction_sum: i64 = 0;
            let mut weight_sum: f64 = 0.0;

            for &neighbor_id in &neighbors {
                let Some(twist) = self.lut.latest(node_id, neighbor_id)
                else {
                    continue;
                };
                let Some(neighbor_frame) = self.nodes.get(&neighbor_id)
                else {
                    continue;
                };

                // Edge direction: the twist is stored for the canonical
                // edge key (smaller_id, larger_id). The twist offset means
                // larger_id.offset = smaller_id.offset + twist.offset.
                // So if we are the larger id: expected = neighbor + twist
                //    if we are the smaller id: expected = neighbor - twist
                let expected = if node_id.0 > neighbor_id.0 {
                    neighbor_frame.offset_nanos.saturating_add(twist.offset_nanos)
                } else {
                    neighbor_frame.offset_nanos.saturating_sub(twist.offset_nanos)
                };
                let error = expected.saturating_sub(node_frame.offset_nanos);

                let weight = twist.quality as f64 / 100.0;
                correction_sum =
                    correction_sum.saturating_add((error as f64 * weight) as i64);
                weight_sum += weight;

                defects.push(error.unsigned_abs());
            }

            if weight_sum > 0.0 {
                let correction =
                    (correction_sum as f64 / weight_sum * self.damping) as i64;
                updates.insert(node_id, correction);
            }
        }

        for (id, correction) in &updates {
            if let Some(frame) = self.nodes.get_mut(id) {
                frame.offset_nanos = frame.offset_nanos.saturating_add(*correction);
            }
        }

        self.iteration += 1;

        let max_defect = defects.iter().copied().max().unwrap_or(0);
        let mean_defect = if defects.is_empty() {
            0
        } else {
            defects.iter().sum::<u64>() / defects.len() as u64
        };

        SolveResult {
            iterations: self.iteration,
            max_defect_nanos: max_defect,
            mean_defect_nanos: mean_defect,
            converged: max_defect < 1000, // < 1μs
        }
    }

    /// Run the spatial expansion step: try multiple resolution
    /// attempts in parallel and keep the best one.
    /// This is the mechanism for ejecting bad nodes.
    pub fn step_spatial_expansion(&mut self, attempts: usize) -> SolveResult {
        let mut best_result: Option<(SolveResult, HashMap<NodeId, ChiralFrame>)> = None;

        for _ in 0..attempts {
            let saved_nodes = self.nodes.clone();
            let result = self.step();

            let is_better = match &best_result {
                None => true,
                Some((best, _)) => {
                    result.max_defect_nanos < best.max_defect_nanos
                }
            };

            if is_better {
                best_result = Some((result, self.nodes.clone()));
            }

            self.nodes = saved_nodes;
        }

        if let Some((result, best_nodes)) = best_result {
            self.nodes = best_nodes;
            self.iteration += 1;
            result
        } else {
            SolveResult {
                iterations: self.iteration,
                max_defect_nanos: u64::MAX,
                mean_defect_nanos: u64::MAX,
                converged: false,
            }
        }
    }

    /// Run until convergence or max iterations.
    pub fn solve(&mut self, max_iterations: u32, tolerance_nanos: u64) -> SolveResult {
        let mut result = SolveResult {
            iterations: 0,
            max_defect_nanos: u64::MAX,
            mean_defect_nanos: u64::MAX,
            converged: false,
        };

        for _ in 0..max_iterations {
            result = self.step();
            if result.max_defect_nanos <= tolerance_nanos {
                result.converged = true;
                return result;
            }
        }

        result
    }

    /// Detect nodes with high holonomy defect (Byzantine / broken clocks).
    pub fn detect_defective_nodes(&self, threshold_nanos: u64) -> Vec<NodeId> {
        let mut defective = Vec::new();

        for (&node_id, node_frame) in &self.nodes {
            let neighbors = self.lut.neighbors(node_id);
            if neighbors.len() < 2 {
                continue;
            }

            let mut total_defect: u64 = 0;
            let mut count: u32 = 0;

            for &neighbor_id in &neighbors {
                let Some(twist) = self.lut.latest(node_id, neighbor_id)
                else {
                    continue;
                };
                let Some(neighbor_frame) = self.nodes.get(&neighbor_id)
                else {
                    continue;
                };

                let expected = if node_id.0 > neighbor_id.0 {
                    neighbor_frame.offset_nanos.saturating_add(twist.offset_nanos)
                } else {
                    neighbor_frame.offset_nanos.saturating_sub(twist.offset_nanos)
                };
                let error = expected.saturating_sub(node_frame.offset_nanos).unsigned_abs();
                total_defect += error;
                count += 1;
            }

            if count > 0 && total_defect / count as u64 > threshold_nanos {
                defective.push(node_id);
            }
        }

        defective
    }
}

impl Default for HolonomySolver {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::holonomy::twist::TwistIndex;

    fn make_twist(offset_nanos: i64, rtt_nanos: u64) -> TwistIndex {
        TwistIndex {
            rtt_nanos,
            offset_nanos,
            asymmetry_nanos: 0,
            quality: 100,
            epoch_measured: 1,
        }
    }

    #[test]
    fn test_solver_converges_line() {
        // Three nodes in a line: A --[+1ms]--> B --[+2ms]--> C
        let mut solver = HolonomySolver::new();
        let a = NodeId([1u8; 32]);
        let b = NodeId([2u8; 32]);
        let c = NodeId([3u8; 32]);

        solver.add_node(a, ChiralFrame::new(0, 0, 100));
        solver.add_node(b, ChiralFrame::new(0, 0, 100));
        solver.add_node(c, ChiralFrame::new(0, 0, 100));

        solver.add_measurement(a, b, make_twist(1_000_000, 10_000_000));
        solver.add_measurement(b, c, make_twist(2_000_000, 10_000_000));

        let result = solver.solve(500, 1_000);
        assert!(
            result.converged,
            "did not converge: max_defect={}ns in {} iterations",
            result.max_defect_nanos, result.iterations
        );
    }

    #[test]
    fn test_solver_detects_byzantine() {
        // 4 honest nodes + 1 Byzantine claiming a wildly wrong offset
        let mut solver = HolonomySolver::new();
        let nodes: Vec<NodeId> = (0..5)
            .map(|i| NodeId([i as u8; 32]))
            .collect();

        // Honest nodes all at ~0 offset
        for i in 0..4 {
            solver.add_node(nodes[i], ChiralFrame::new(0, 0, 100));
        }
        // Byzantine node claims +1 second offset
        solver.add_node(nodes[4], ChiralFrame::new(1_000_000_000, 0, 100));

        // Fully connected measurement graph with consistent twists
        for i in 0..5 {
            for j in (i + 1)..5 {
                // Twist based on honest offsets (near zero between honest nodes)
                let twist_offset = if i < 4 && j < 4 {
                    0 // honest-honest
                } else if j == 4 {
                    1_000_000 // honest-byzantine (measured correctly)
                } else {
                    0
                };
                solver.add_measurement(
                    nodes[i],
                    nodes[j],
                    make_twist(twist_offset, 10_000_000),
                );
            }
        }

        let defective = solver.detect_defective_nodes(100_000_000);
        assert!(
            defective.contains(&nodes[4]),
            "should detect Byzantine node"
        );
    }
}
