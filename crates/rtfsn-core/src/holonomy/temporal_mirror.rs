use serde::{Deserialize, Serialize};

/// TemporalMirror: retrodictive-predictive cascade for finding
/// the fixed-point time value at each epoch boundary.
///
/// Mirror A (predictive): extrapolate forward from current state
/// Mirror B (retrodictive): work backwards from the epoch boundary
///
/// The fixed point is where both mirrors agree — a self-consistent
/// time value that is the same regardless of temporal direction.

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalMirror {
    pub predictive_time_nanos: u64,
    pub retrodictive_time_nanos: u64,
    pub fixed_point_nanos: u64,
    pub disagreement_nanos: u64,
    pub iterations_to_converge: u32,
}

impl TemporalMirror {
    /// Find the fixed point between predictive and retrodictive estimates.
    ///
    /// `forward_estimates`: clock estimates extrapolated forward
    /// `backward_estimates`: clock estimates retrodicted from boundary
    /// `max_iterations`: convergence limit
    pub fn find_fixed_point(
        forward_estimates: &[i64],
        backward_estimates: &[i64],
        max_iterations: u32,
    ) -> Self {
        if forward_estimates.is_empty() || backward_estimates.is_empty() {
            return Self {
                predictive_time_nanos: 0,
                retrodictive_time_nanos: 0,
                fixed_point_nanos: 0,
                disagreement_nanos: u64::MAX,
                iterations_to_converge: 0,
            };
        }

        // Weighted mean of forward estimates
        let fwd_mean =
            forward_estimates.iter().sum::<i64>() / forward_estimates.len() as i64;
        let bwd_mean =
            backward_estimates.iter().sum::<i64>() / backward_estimates.len() as i64;

        let mut predictive = fwd_mean;
        let mut retrodictive = bwd_mean;

        let mut iterations = 0;
        for i in 0..max_iterations {
            iterations = i + 1;

            // Bisect toward agreement
            let midpoint = (predictive + retrodictive) / 2;

            // Adjust each mirror toward midpoint with damping
            let alpha = 0.5;
            predictive = predictive
                + ((midpoint - predictive) as f64 * alpha) as i64;
            retrodictive = retrodictive
                + ((midpoint - retrodictive) as f64 * alpha) as i64;

            let disagreement = (predictive - retrodictive).unsigned_abs();
            if disagreement < 1000 {
                // < 1μs
                break;
            }
        }

        let fixed_point = (predictive + retrodictive) / 2;
        let disagreement = (predictive - retrodictive).unsigned_abs();

        Self {
            predictive_time_nanos: predictive as u64,
            retrodictive_time_nanos: retrodictive as u64,
            fixed_point_nanos: fixed_point as u64,
            disagreement_nanos: disagreement,
            iterations_to_converge: iterations,
        }
    }

    pub fn is_consistent(&self, tolerance_nanos: u64) -> bool {
        self.disagreement_nanos <= tolerance_nanos
    }
}

/// Epoch boundary solver using the temporal mirror.
#[derive(Debug)]
pub struct EpochBoundarySolver {
    pub epoch_duration_nanos: u64,
    mirrors: Vec<TemporalMirror>,
}

impl EpochBoundarySolver {
    pub fn new(epoch_duration_nanos: u64) -> Self {
        Self {
            epoch_duration_nanos,
            mirrors: Vec::new(),
        }
    }

    /// Solve for the time at epoch boundary N, given:
    /// - `node_offsets`: each node's estimated offset at this epoch
    /// - `node_drifts_ppb`: each node's drift rate
    /// - `prev_time_nanos`: the solved time at epoch N-1
    pub fn solve_boundary(
        &mut self,
        node_offsets: &[i64],
        node_drifts_ppb: &[i64],
        prev_time_nanos: u64,
    ) -> &TemporalMirror {
        let dt = self.epoch_duration_nanos as i64;

        // Forward: predict from previous boundary + drift
        let forward: Vec<i64> = node_offsets
            .iter()
            .zip(node_drifts_ppb.iter())
            .map(|(&offset, &drift)| {
                prev_time_nanos as i64 + dt + offset
                    + (drift * dt) / 1_000_000_000
            })
            .collect();

        // Backward: retrodict from the next boundary
        // (estimate where the boundary "should" be given the offsets)
        let backward: Vec<i64> = node_offsets
            .iter()
            .map(|&offset| {
                prev_time_nanos as i64 + dt + offset
            })
            .collect();

        let mirror =
            TemporalMirror::find_fixed_point(&forward, &backward, 50);
        self.mirrors.push(mirror);
        self.mirrors.last().unwrap()
    }

    pub fn latest_mirror(&self) -> Option<&TemporalMirror> {
        self.mirrors.last()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temporal_mirror_convergence() {
        let forward = vec![1_000_000_000, 1_000_001_000, 999_999_000];
        let backward = vec![1_000_000_500, 1_000_000_200, 999_999_800];

        let mirror =
            TemporalMirror::find_fixed_point(&forward, &backward, 100);
        assert!(
            mirror.is_consistent(1_000),
            "disagreement={}",
            mirror.disagreement_nanos
        );
    }

    #[test]
    fn test_epoch_boundary_solver() {
        let mut solver = EpochBoundarySolver::new(30_000_000_000); // 30s epochs

        let offsets = vec![100, -200, 50, -80, 150];
        let drifts = vec![10, -5, 3, -2, 8];

        let mirror = solver.solve_boundary(&offsets, &drifts, 0);
        assert!(mirror.is_consistent(10_000));
    }
}
