use serde::{Deserialize, Serialize};

/// NTP-like pairwise time exchange result.
///
/// Given a 4-timestamp exchange (t1, t2, t3, t4):
///   offset = ((t2 - t1) + (t3 - t4)) / 2
///   delay  = (t4 - t1) - (t3 - t2)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeExchange {
    pub t1: f64, // client send
    pub t2: f64, // server receive
    pub t3: f64, // server send
    pub t4: f64, // client receive
}

impl TimeExchange {
    pub fn new(t1: f64, t2: f64, t3: f64, t4: f64) -> Self {
        Self { t1, t2, t3, t4 }
    }

    pub fn offset(&self) -> f64 {
        ((self.t2 - self.t1) + (self.t3 - self.t4)) / 2.0
    }

    pub fn round_trip_delay(&self) -> f64 {
        (self.t4 - self.t1) - (self.t3 - self.t2)
    }

    pub fn asymmetry_bound(&self) -> f64 {
        self.round_trip_delay() / 2.0
    }
}

/// Tracks clock offset and drift rate for a node relative to network time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClockState {
    pub offset: f64,
    pub drift_rate: f64,
    pub uncertainty: f64,
    pub last_update: f64,
    samples: Vec<ClockSample>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ClockSample {
    time: f64,
    offset: f64,
    weight: f64,
}

impl ClockState {
    pub fn new() -> Self {
        Self {
            offset: 0.0,
            drift_rate: 0.0,
            uncertainty: f64::MAX,
            last_update: 0.0,
            samples: Vec::new(),
        }
    }

    pub fn add_exchange(&mut self, exchange: &TimeExchange, local_time: f64) {
        let delay = exchange.round_trip_delay();
        if delay <= 0.0 {
            return;
        }

        let weight = 1.0 / delay;
        self.samples.push(ClockSample {
            time: local_time,
            offset: exchange.offset(),
            weight,
        });

        const MAX_SAMPLES: usize = 64;
        if self.samples.len() > MAX_SAMPLES {
            self.samples.remove(0);
        }

        self.recompute();
        self.last_update = local_time;
    }

    fn recompute(&mut self) {
        if self.samples.is_empty() {
            return;
        }

        if self.samples.len() == 1 {
            self.offset = self.samples[0].offset;
            self.drift_rate = 0.0;
            self.uncertainty = 1.0 / self.samples[0].weight;
            return;
        }

        // Weighted linear regression: offset = a + b*time
        let total_weight: f64 = self.samples.iter().map(|s| s.weight).sum();
        let mean_t: f64 =
            self.samples.iter().map(|s| s.weight * s.time).sum::<f64>()
                / total_weight;
        let mean_o: f64 = self
            .samples
            .iter()
            .map(|s| s.weight * s.offset)
            .sum::<f64>()
            / total_weight;

        let mut ss_tt = 0.0;
        let mut ss_to = 0.0;

        for s in &self.samples {
            let dt = s.time - mean_t;
            let do_ = s.offset - mean_o;
            ss_tt += s.weight * dt * dt;
            ss_to += s.weight * dt * do_;
        }

        if ss_tt.abs() > 1e-20 {
            self.drift_rate = ss_to / ss_tt;
            self.offset = mean_o - self.drift_rate * mean_t;
        } else {
            self.drift_rate = 0.0;
            self.offset = mean_o;
        }

        // Residual-based uncertainty
        let mut ss_residual = 0.0;
        for s in &self.samples {
            let predicted = self.offset + self.drift_rate * s.time;
            let residual = s.offset - predicted;
            ss_residual += s.weight * residual * residual;
        }
        self.uncertainty =
            (ss_residual / total_weight).sqrt().max(1e-9);
    }

    pub fn predict(&self, time: f64) -> f64 {
        self.offset + self.drift_rate * time
    }
}

impl Default for ClockState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_time_exchange_offset() {
        // Symmetric path: offset should be 1.0
        let ex = TimeExchange::new(0.0, 1.1, 1.1, 0.2);
        let offset = ex.offset();
        let delay = ex.round_trip_delay();
        assert!((offset - 1.0).abs() < 0.01, "offset = {offset}");
        assert!(delay > 0.0);
    }

    #[test]
    fn test_clock_state_linear_drift() {
        let mut state = ClockState::new();

        // Simulate a clock that drifts at 100 ppm (0.0001 s/s)
        for i in 0..20 {
            let t = i as f64 * 10.0;
            let true_offset = 0.5 + 0.0001 * t;

            let ex = TimeExchange::new(
                t,
                t + true_offset + 0.005,
                t + true_offset + 0.005,
                t + 0.01,
            );
            state.add_exchange(&ex, t);
        }

        let predicted = state.predict(200.0);
        let expected = 0.5 + 0.0001 * 200.0;
        assert!(
            (predicted - expected).abs() < 0.05,
            "predicted={predicted}, expected={expected}"
        );
    }
}
