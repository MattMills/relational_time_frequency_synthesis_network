use serde::{Deserialize, Serialize};

/// Marzullo's algorithm for intersecting time confidence intervals.
/// Given a set of (offset, uncertainty) pairs from different sources,
/// find the tightest interval consistent with a majority.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeInterval {
    pub low: f64,
    pub high: f64,
}

impl TimeInterval {
    pub fn from_measurement(offset: f64, uncertainty: f64) -> Self {
        Self {
            low: offset - uncertainty,
            high: offset + uncertainty,
        }
    }

    pub fn midpoint(&self) -> f64 {
        (self.low + self.high) / 2.0
    }

    pub fn width(&self) -> f64 {
        self.high - self.low
    }
}

pub fn marzullo(intervals: &[TimeInterval]) -> Option<TimeInterval> {
    if intervals.is_empty() {
        return None;
    }

    #[derive(PartialEq)]
    enum EventType {
        Start,
        End,
    }

    let mut events: Vec<(f64, EventType)> = Vec::with_capacity(intervals.len() * 2);
    for iv in intervals {
        events.push((iv.low, EventType::Start));
        events.push((iv.high, EventType::End));
    }

    events.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                // Start events before End events at the same point
                match (&a.1, &b.1) {
                    (EventType::Start, EventType::End) => {
                        std::cmp::Ordering::Less
                    }
                    (EventType::End, EventType::Start) => {
                        std::cmp::Ordering::Greater
                    }
                    _ => std::cmp::Ordering::Equal,
                }
            })
    });

    let n = intervals.len();
    let threshold = (n / 2) + 1; // strict majority

    let mut best_start = f64::NAN;
    let mut best_end = f64::NAN;
    let mut best_count = 0usize;
    let mut count = 0usize;
    let mut current_start = f64::NAN;

    for (point, event_type) in &events {
        match event_type {
            EventType::Start => {
                count += 1;
                if count >= threshold && count > best_count {
                    current_start = *point;
                }
            }
            EventType::End => {
                if count >= threshold && count > best_count {
                    best_start = current_start;
                    best_end = *point;
                    best_count = count;
                }
                count -= 1;
            }
        }
    }

    if best_count >= threshold {
        Some(TimeInterval {
            low: best_start,
            high: best_end,
        })
    } else {
        None
    }
}

/// Geometric consistency check: given a set of offset estimates and
/// their predicted vs measured RTTs, reject nodes whose offset is
/// inconsistent with the latency geometry.
pub fn reject_geometric_outliers(
    offsets: &[(f64, f64)], // (offset, residual)
    max_residual: f64,
) -> Vec<usize> {
    let filtered: Vec<f64> = offsets
        .iter()
        .filter(|(_, r)| *r <= max_residual)
        .map(|(o, _)| *o)
        .collect();

    if filtered.is_empty() {
        return Vec::new();
    }

    let mean: f64 = filtered.iter().sum::<f64>() / filtered.len() as f64;
    let variance: f64 = filtered.iter().map(|o| (o - mean).powi(2)).sum::<f64>()
        / filtered.len() as f64;
    let stddev = variance.sqrt();

    // Reject if either high geometric residual OR statistical outlier
    let threshold = 2.5 * stddev;
    offsets
        .iter()
        .enumerate()
        .filter(|(_, (offset, residual))| {
            *residual > max_residual || (offset - mean).abs() > threshold
        })
        .map(|(i, _)| i)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_marzullo_basic() {
        let intervals = vec![
            TimeInterval::from_measurement(1.0, 0.1),
            TimeInterval::from_measurement(1.05, 0.1),
            TimeInterval::from_measurement(1.02, 0.1),
            TimeInterval::from_measurement(5.0, 0.1), // outlier
        ];

        let result = marzullo(&intervals).unwrap();
        assert!(result.low >= 0.9);
        assert!(result.high <= 1.15);
        assert!(result.width() > 0.0);
    }

    #[test]
    fn test_geometric_outlier_rejection() {
        let offsets = vec![
            (1.0, 0.001),   // good
            (1.01, 0.002),  // good
            (0.99, 0.001),  // good
            (1.02, 0.003),  // good
            (5.0, 0.5),     // bad: high residual
            (1.0, 0.001),   // good
        ];

        let rejected = reject_geometric_outliers(&offsets, 0.01);
        assert!(rejected.contains(&4), "should reject index 4");
        assert!(!rejected.contains(&0), "should keep index 0");
    }
}
