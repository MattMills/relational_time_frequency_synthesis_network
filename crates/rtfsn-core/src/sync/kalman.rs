use serde::{Deserialize, Serialize};

/// 2-state Kalman filter tracking clock offset and drift.
///
/// State vector: [offset, drift_rate]
/// The drift rate is modeled as a random walk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClockKalman {
    // State: [offset, drift_rate]
    pub x: [f64; 2],
    // Covariance matrix (symmetric 2x2, stored as [p00, p01, p11])
    pub p: [f64; 3],
    // Process noise
    pub q_offset: f64,
    pub q_drift: f64,
}

impl ClockKalman {
    pub fn new() -> Self {
        Self {
            x: [0.0, 0.0],
            p: [1.0, 0.0, 1e-6],
            q_offset: 1e-8,
            q_drift: 1e-12,
        }
    }

    pub fn offset(&self) -> f64 {
        self.x[0]
    }

    pub fn drift_rate(&self) -> f64 {
        self.x[1]
    }

    pub fn uncertainty(&self) -> f64 {
        self.p[0].sqrt()
    }

    /// Predict step: advance the state by dt seconds.
    pub fn predict(&mut self, dt: f64) {
        // State transition: offset += drift_rate * dt
        self.x[0] += self.x[1] * dt;

        // Covariance propagation: P = F*P*F' + Q
        // F = [[1, dt], [0, 1]]
        let p00 = self.p[0] + 2.0 * dt * self.p[1]
            + dt * dt * self.p[2]
            + self.q_offset * dt;
        let p01 = self.p[1] + dt * self.p[2];
        let p11 = self.p[2] + self.q_drift * dt;

        self.p = [p00, p01, p11];
    }

    /// Update step: incorporate a new offset measurement with given variance.
    pub fn update(&mut self, measured_offset: f64, measurement_variance: f64) {
        // Measurement model: z = H*x + v, H = [1, 0]
        let innovation = measured_offset - self.x[0];

        // Innovation covariance: S = H*P*H' + R
        let s = self.p[0] + measurement_variance;
        if s.abs() < 1e-30 {
            return;
        }

        // Kalman gain: K = P*H'/S
        let k0 = self.p[0] / s;
        let k1 = self.p[1] / s;

        // State update
        self.x[0] += k0 * innovation;
        self.x[1] += k1 * innovation;

        // Covariance update (Joseph form for numerical stability)
        let p00 = (1.0 - k0) * self.p[0];
        let p01 = (1.0 - k0) * self.p[1];
        let p11 = self.p[2] - k1 * self.p[1];

        self.p = [p00, p01, p11];
    }

    /// Combined predict + update for a measurement at the given time delta.
    pub fn step(
        &mut self,
        dt: f64,
        measured_offset: f64,
        measurement_variance: f64,
    ) {
        self.predict(dt);
        self.update(measured_offset, measurement_variance);
    }
}

impl Default for ClockKalman {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kalman_converges_on_constant_offset() {
        let mut kf = ClockKalman::new();
        let true_offset = 0.042;

        for i in 0..50 {
            let dt = 1.0;
            let noise = if i % 2 == 0 { 0.001 } else { -0.001 };
            kf.step(dt, true_offset + noise, 0.001);
        }

        assert!(
            (kf.offset() - true_offset).abs() < 0.005,
            "offset={}, expected={}",
            kf.offset(),
            true_offset
        );
        assert!(
            kf.drift_rate().abs() < 1e-4,
            "drift={}, expected ~0",
            kf.drift_rate()
        );
    }

    #[test]
    fn test_kalman_tracks_drift() {
        let mut kf = ClockKalman::new();
        let drift_rate = 0.0001; // 100 ppm

        for i in 0..200 {
            let t = i as f64;
            let true_offset = 0.01 + drift_rate * t;
            kf.step(1.0, true_offset, 0.0001);
        }

        assert!(
            (kf.drift_rate() - drift_rate).abs() < 1e-5,
            "drift={}, expected={}",
            kf.drift_rate(),
            drift_rate
        );
    }
}
