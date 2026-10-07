//! The relational clock: offset and relative frequency between free-running oscillators, per
//! peer and solved over the whole measured graph.
//!
//! Sans-IO. The caller supplies the four nanosecond timestamps of each exchange — `t1` and `t4`
//! on the initiator's oscillator, `t2` and `t3` on the responder's — and the edge estimates other
//! nodes gossip. Nothing here reads a clock, so a free-running (un-disciplined) oscillator is what
//! the caller should stamp with: its rate against every peer's is what this measures.
//!
//! Three layers:
//!
//! * [`PeerTrack`]: one peer. Each exchange becomes a [`TwistIndex`]. An exchange whose round trip
//!   sits well above the windowed floor is rejected, since a queue on one leg biases the offset by
//!   up to half the excess. The rest feed a [`ClockKalman`] whose state is the offset and the
//!   *relative frequency* (the peer's rate over ours, minus one), with a measurement variance of
//!   the stamping noise plus half the excess round trip, squared.
//! * [`RelationalClock`]: one node. Its own tracks, the edges it has heard from the others, and
//!   the raw measurements in a bounded [`TwistLUT`].
//! * [`solve_frame`]: the network frame. Relative frequencies first, then offsets at one instant,
//!   each by weighted least squares on the graph Laplacian with the reference node held at zero
//!   (the gauge). Every edge's residual, and the holonomy of every fundamental cycle against the
//!   spread its edges' own uncertainties predict for it.
//!
//! What a cycle can and cannot see: path asymmetry, a bad timestamp, a stale edge, or a node that
//! tells different peers different things all leave holonomy. A node that shifts *every* one of
//! its timestamps consistently is a coboundary: the frame absorbs it as that node's offset and no
//! cycle moves. That is cliff-time's forgery result; the README's "a bad clock is detectable from
//! any vantage point" holds for inconsistent clocks only.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use serde::{Deserialize, Serialize};

use crate::holonomy::chiral::ChiralFrame;
use crate::holonomy::twist::{TwistIndex, TwistLUT};
use crate::sync::kalman::ClockKalman;
use crate::types::NodeId;

/// Tuning for a [`PeerTrack`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TrackConfig {
    /// Standard deviation of one exchange's offset reading with no queueing, in ns: the stamping
    /// jitter of the two hosts.
    pub stamp_noise_ns: f64,
    /// Offset process noise, s²/s (white frequency noise of the pair of oscillators).
    pub q_offset: f64,
    /// Relative-frequency process noise, 1/s (random-walk frequency: temperature, ageing).
    pub q_drift: f64,
    /// Prior standard deviation of the relative frequency, before any measurement.
    pub prior_drift: f64,
    /// Round trips kept for the windowed floor.
    pub floor_window: usize,
    /// An exchange whose round trip exceeds the floor by more than
    /// `max(popcorn_ns, popcorn_ratio · floor)` is rejected, once `warmup` have been accepted.
    pub popcorn_ns: f64,
    /// See [`popcorn_ns`](Self::popcorn_ns).
    pub popcorn_ratio: f64,
    /// Exchanges accepted unconditionally while the floor settles.
    pub warmup: u64,
    /// Raw measurements kept per edge in the [`TwistLUT`].
    pub history: usize,
    /// An offset this far (ns) from the track's prediction, and more than fifty of its own
    /// standard deviations, is a step: the peer's timescale moved (a restart that re-anchored
    /// it, a clock set by hand). One is rejected; a second in a row starts the track afresh.
    #[serde(default = "default_step_ns")]
    pub step_ns: f64,
}

fn default_step_ns() -> f64 {
    5_000_000.0
}

impl Default for TrackConfig {
    fn default() -> Self {
        TrackConfig {
            stamp_noise_ns: 20_000.0,
            q_offset: 1e-16,
            q_drift: 1e-18,
            prior_drift: 1e-4,
            floor_window: 64,
            popcorn_ns: 100_000.0,
            popcorn_ratio: 0.5,
            warmup: 8,
            history: 256,
            step_ns: default_step_ns(),
        }
    }
}

/// One node's relation to one peer at an instant of its own clock.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Relation {
    /// The instant, on the measuring node's clock, in ns.
    pub at_ns: u64,
    /// Peer's clock minus ours at `at_ns`, in ns.
    pub offset_ns: f64,
    /// Relative frequency: the peer's rate over ours, minus one (1e-6 is one ppm).
    pub drift: f64,
    /// Standard deviation of `offset_ns`.
    pub sigma_offset_ns: f64,
    /// Standard deviation of `drift`.
    pub sigma_drift: f64,
}

impl Relation {
    /// The offset extrapolated to `at_ns` on the measuring node's clock.
    pub fn offset_at(&self, at_ns: f64) -> f64 {
        self.offset_ns + self.drift * (at_ns - self.at_ns as f64)
    }

    /// The offset's standard deviation extrapolated to `at_ns` (covariance ignored).
    pub fn sigma_at(&self, at_ns: f64) -> f64 {
        let age = at_ns - self.at_ns as f64;
        (self.sigma_offset_ns.powi(2) + (self.sigma_drift * age).powi(2)).sqrt()
    }
}

/// Why an exchange was not used.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Reject {
    /// The round trip was zero or negative: the timestamps are inconsistent.
    NonPositiveRtt,
    /// The round trip exceeded the floor by this much: a queue on one leg.
    Popcorn {
        /// Excess over the windowed floor, ns.
        excess_ns: u64,
    },
    /// Older than an exchange already absorbed.
    OutOfOrder,
    /// The offset jumped from the prediction by this much (ns); a second jump in a row restarts
    /// the track.
    Step {
        /// Measured minus predicted offset, ns.
        jump_ns: i64,
    },
}

/// The outcome of one exchange.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Observation {
    /// Absorbed; the relation after it.
    Accepted {
        /// The raw measurement.
        twist: TwistIndex,
        /// The track's relation after absorbing it.
        relation: Relation,
        /// The track was started afresh on this exchange after a step.
        restarted: bool,
    },
    /// Not used.
    Rejected {
        /// Why.
        reason: Reject,
    },
}

/// What one node knows of one peer's oscillator relative to its own, from the exchanges it
/// initiated.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerTrack {
    cfg: TrackConfig,
    /// Our-clock instant of the first accepted exchange, and its offset; the Kalman state is
    /// relative to these so it stays small.
    anchor_ns: u64,
    base_offset_ns: i64,
    last_ns: u64,
    kalman: ClockKalman,
    rtts: VecDeque<u64>,
    /// Exchanges absorbed.
    pub accepted: u64,
    /// Exchanges rejected.
    pub rejected: u64,
    /// The last absorbed measurement.
    pub last: Option<TwistIndex>,
    /// Times the track restarted after a step.
    #[serde(default)]
    pub steps: u64,
    #[serde(default)]
    suspect: u32,
}

impl PeerTrack {
    /// An empty track.
    pub fn new(cfg: TrackConfig) -> Self {
        PeerTrack {
            cfg,
            anchor_ns: 0,
            base_offset_ns: 0,
            last_ns: 0,
            kalman: ClockKalman::new(),
            rtts: VecDeque::new(),
            accepted: 0,
            rejected: 0,
            last: None,
            steps: 0,
            suspect: 0,
        }
    }

    /// The windowed minimum round trip, in ns.
    pub fn rtt_floor_ns(&self) -> Option<u64> {
        self.rtts.iter().min().copied()
    }

    /// Our-clock instant of the first absorbed exchange.
    pub fn anchor_ns(&self) -> u64 {
        self.anchor_ns
    }

    /// Absorb one exchange this node initiated: `t1`, `t4` on our clock, `t2`, `t3` on the peer's.
    pub fn observe(&mut self, t1: u64, t2: u64, t3: u64, t4: u64) -> Observation {
        let rtt = (t4 as i128 - t1 as i128) - (t3 as i128 - t2 as i128);
        if rtt <= 0 || t4 < t1 {
            self.rejected += 1;
            return Observation::Rejected {
                reason: Reject::NonPositiveRtt,
            };
        }
        let rtt = rtt as u64;
        let mid = t1 + (t4 - t1) / 2;
        if self.accepted > 0 && mid <= self.last_ns {
            self.rejected += 1;
            return Observation::Rejected {
                reason: Reject::OutOfOrder,
            };
        }
        self.rtts.push_back(rtt);
        while self.rtts.len() > self.cfg.floor_window.max(1) {
            self.rtts.pop_front();
        }
        let floor = self.rtt_floor_ns().unwrap_or(rtt);
        let excess = (rtt - floor) as f64;
        let limit = self
            .cfg
            .popcorn_ns
            .max(self.cfg.popcorn_ratio * floor as f64);
        if self.accepted >= self.cfg.warmup && excess > limit {
            self.rejected += 1;
            return Observation::Rejected {
                reason: Reject::Popcorn {
                    excess_ns: excess as u64,
                },
            };
        }
        let twist = TwistIndex::from_exchange(t1, t2, t3, t4);
        let mut restarted = false;
        if self.accepted >= self.cfg.warmup {
            let dt = (mid - self.last_ns) as f64 * 1e-9;
            let predicted =
                self.base_offset_ns as f64 + (self.kalman.x[0] + self.kalman.x[1] * dt) * 1e9;
            let jump = twist.offset_nanos as f64 - predicted;
            let spread =
                (self.kalman.p[0] + 2.0 * dt * self.kalman.p[1] + dt * dt * self.kalman.p[2])
                    .max(0.0)
                    .sqrt()
                    * 1e9
                    + self.cfg.stamp_noise_ns
                    + excess / 2.0;
            if jump.abs() > self.cfg.step_ns.max(50.0 * spread) {
                self.suspect += 1;
                if self.suspect < 2 {
                    self.rejected += 1;
                    return Observation::Rejected {
                        reason: Reject::Step {
                            jump_ns: jump as i64,
                        },
                    };
                }
                // Twice in a row: the peer's timescale moved. Start over from this exchange.
                self.steps += 1;
                self.accepted = 0;
                self.rtts.clear();
                self.rtts.push_back(rtt);
                restarted = true;
            }
            self.suspect = 0;
        }
        // Until the floor has settled it cannot say how much of a round trip is queue (an early
        // spike would measure its excess against itself), so the bound is the whole asymmetry
        // half a round trip allows.
        let bias_bound = if self.accepted < self.cfg.warmup {
            rtt as f64 / 2.0
        } else {
            excess / 2.0
        };
        let sigma_s = (self.cfg.stamp_noise_ns + bias_bound) * 1e-9;
        let r = sigma_s * sigma_s;
        if self.accepted == 0 {
            self.anchor_ns = mid;
            self.base_offset_ns = twist.offset_nanos;
            self.kalman = ClockKalman {
                x: [0.0, 0.0],
                p: [r, 0.0, self.cfg.prior_drift * self.cfg.prior_drift],
                q_offset: self.cfg.q_offset,
                q_drift: self.cfg.q_drift,
            };
        } else {
            let dt = (mid - self.last_ns) as f64 * 1e-9;
            let measured = (twist.offset_nanos - self.base_offset_ns) as f64 * 1e-9;
            self.kalman.step(dt, measured, r);
        }
        self.last_ns = mid;
        self.accepted += 1;
        self.last = Some(twist.clone());
        Observation::Accepted {
            twist,
            relation: self.relation().expect("just absorbed one"),
            restarted,
        }
    }

    /// The relation as of the last absorbed exchange.
    pub fn relation(&self) -> Option<Relation> {
        if self.accepted == 0 {
            return None;
        }
        Some(Relation {
            at_ns: self.last_ns,
            offset_ns: self.base_offset_ns as f64 + self.kalman.x[0] * 1e9,
            drift: self.kalman.x[1],
            sigma_offset_ns: self.kalman.p[0].max(0.0).sqrt() * 1e9,
            sigma_drift: self.kalman.p[2].max(0.0).sqrt(),
        })
    }
}

/// One directed edge of the measured graph: what `from` knows of `to`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EdgeEstimate {
    /// The measuring node.
    pub from: NodeId,
    /// The measured node.
    pub to: NodeId,
    /// `to` relative to `from`, on `from`'s clock.
    pub relation: Relation,
    /// The windowed minimum round trip, ns.
    pub rtt_floor_ns: u64,
    /// Exchanges behind the estimate.
    pub samples: u64,
}

struct Heard {
    edge: EdgeEstimate,
    received_ns: u64,
}

/// The node that stands for UTC itself in an absolute frame. It neither measures nor is measured:
/// [`RelationalClock::solve_absolute`] joins it to every [`Anchor`] by an edge whose spread is the
/// anchor's distance from UTC, and holds it at zero.
pub const UTC: NodeId = NodeId([0xFF; 32]);

/// An absolute reference: a node (usually an NTP server) whose clock is UTC up to its own error.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Anchor {
    /// The reference.
    pub node: NodeId,
    /// How far its clock may be from UTC, ns: an NTP server's root distance, plus half the round
    /// trip of the path it was measured over (an asymmetry no averaging removes).
    pub sigma_offset_ns: f64,
    /// How far its rate may be from UTC's.
    pub sigma_drift: f64,
    /// Its NTP stratum (1 for a primary reference).
    pub stratum: u8,
    /// How long before this copy was handed on the reference was last measured, ns (0 straight
    /// from a measurement). Gossip carries it, so an anchor passed around between nodes ages from
    /// its last real measurement instead of looking fresh at every hop -- otherwise a server
    /// nobody queries any more would circulate for ever.
    #[serde(default)]
    pub age_ns: u64,
}

struct HeldAnchor {
    anchor: Anchor,
    /// When the reference was last measured, on this node's clock.
    measured_ns: u64,
    own: bool,
}

/// One node of a relational clock network: its own tracks and what it has heard.
pub struct RelationalClock {
    me: NodeId,
    cfg: TrackConfig,
    tracks: BTreeMap<NodeId, PeerTrack>,
    heard: BTreeMap<(NodeId, NodeId), Heard>,
    anchors: BTreeMap<NodeId, HeldAnchor>,
    lut: TwistLUT,
    inserts: usize,
}

impl RelationalClock {
    /// A node with no measurements.
    pub fn new(me: NodeId, cfg: TrackConfig) -> Self {
        RelationalClock {
            me,
            cfg,
            tracks: BTreeMap::new(),
            heard: BTreeMap::new(),
            anchors: BTreeMap::new(),
            lut: TwistLUT::new(),
            inserts: 0,
        }
    }

    /// This node.
    pub fn me(&self) -> NodeId {
        self.me
    }

    /// The tuning every new track gets.
    pub fn config(&self) -> TrackConfig {
        self.cfg
    }

    /// Absorb an exchange this node initiated with `peer`.
    pub fn observe(&mut self, peer: NodeId, t1: u64, t2: u64, t3: u64, t4: u64) -> Observation {
        let cfg = self.cfg;
        let track = self
            .tracks
            .entry(peer)
            .or_insert_with(|| PeerTrack::new(cfg));
        let obs = track.observe(t1, t2, t3, t4);
        if let Observation::Accepted { twist, .. } = &obs {
            self.lut.insert(self.me, peer, twist.clone());
            self.inserts += 1;
            if self.inserts.is_multiple_of(64) {
                self.lut.trim(self.cfg.history);
            }
        }
        obs
    }

    /// This node's track of `peer`.
    pub fn track(&self, peer: NodeId) -> Option<&PeerTrack> {
        self.tracks.get(&peer)
    }

    /// Every track.
    pub fn tracks(&self) -> impl Iterator<Item = (NodeId, &PeerTrack)> + '_ {
        self.tracks.iter().map(|(id, t)| (*id, t))
    }

    /// Drop a peer: its track, every heard edge touching it, and its anchor if it is one.
    pub fn forget(&mut self, peer: NodeId) {
        self.tracks.remove(&peer);
        self.heard.retain(|(a, b), _| *a != peer && *b != peer);
        self.anchors.remove(&peer);
    }

    /// Absorb an exchange with an absolute reference (an NTP server answering a request this node
    /// sent): `t1`, `t4` on our clock, `t2`, `t3` on the server's. The server becomes an
    /// [`Anchor`] whose spread is `root_distance_ns` plus half our windowed round trip to it.
    pub fn observe_reference(
        &mut self,
        server: NodeId,
        exchange: (u64, u64, u64, u64),
        root_distance_ns: f64,
        stratum: u8,
        sigma_drift: f64,
        now_ns: u64,
    ) -> Observation {
        let (t1, t2, t3, t4) = exchange;
        let obs = self.observe(server, t1, t2, t3, t4);
        if let Some(t) = self.tracks.get(&server).filter(|t| t.accepted > 0) {
            let floor = t.rtt_floor_ns().unwrap_or(0);
            self.anchors.insert(
                server,
                HeldAnchor {
                    anchor: Anchor {
                        node: server,
                        sigma_offset_ns: root_distance_ns + floor as f64 / 2.0,
                        sigma_drift,
                        stratum,
                        age_ns: 0,
                    },
                    measured_ns: now_ns,
                    own: true,
                },
            );
        }
        obs
    }

    /// Keep an anchor another node measured, heard at `now_ns`; it was measured `anchor.age_ns`
    /// before that. One this node measures itself stays its own; of two for the same reference
    /// the fresher is kept, and of two measured within ten seconds of each other the tighter.
    pub fn hear_anchor(&mut self, anchor: Anchor, now_ns: u64) {
        let measured = now_ns.saturating_sub(anchor.age_ns);
        match self.anchors.get(&anchor.node) {
            Some(h) if h.own => return,
            Some(h) if h.measured_ns > measured + 10_000_000_000 => return,
            Some(h)
                if h.anchor.sigma_offset_ns < anchor.sigma_offset_ns
                    && h.measured_ns + 10_000_000_000 >= measured =>
            {
                return;
            }
            _ => {}
        }
        self.anchors.insert(
            anchor.node,
            HeldAnchor {
                anchor: Anchor { age_ns: 0, ..anchor },
                measured_ns: measured,
                own: false,
            },
        );
    }

    /// Every anchor known, this node's own and heard ones.
    pub fn anchors(&self) -> Vec<Anchor> {
        self.anchors.values().map(|h| h.anchor).collect()
    }

    /// Every anchor known, as handed on at `now_ns`: each carries how long ago it was measured.
    pub fn gossip_anchors(&self, now_ns: u64) -> Vec<Anchor> {
        self.anchors
            .values()
            .map(|h| Anchor {
                age_ns: now_ns.saturating_sub(h.measured_ns),
                ..h.anchor
            })
            .collect()
    }

    /// Forget anchors whose reference nobody has measured for `max_age_ns` (this node's own
    /// included: a server that stopped answering).
    pub fn expire_anchors(&mut self, now_ns: u64, max_age_ns: u64) {
        self.anchors
            .retain(|_, h| now_ns.saturating_sub(h.measured_ns) <= max_age_ns);
    }

    /// The anchors this node measures itself, to gossip.
    pub fn own_anchors(&self) -> Vec<Anchor> {
        self.anchors
            .values()
            .filter(|h| h.own)
            .map(|h| h.anchor)
            .collect()
    }

    /// This node's own edges, one per track with an estimate.
    pub fn edges(&self) -> Vec<EdgeEstimate> {
        self.tracks
            .iter()
            .filter_map(|(&to, t)| {
                Some(EdgeEstimate {
                    from: self.me,
                    to,
                    relation: t.relation()?,
                    rtt_floor_ns: t.rtt_floor_ns().unwrap_or(0),
                    samples: t.accepted,
                })
            })
            .collect()
    }

    /// Keep an edge another node measured, received at `now_ns` on our clock. Our own edges are
    /// ours to measure and are ignored here.
    pub fn hear(&mut self, edge: EdgeEstimate, now_ns: u64) {
        if edge.from == self.me || edge.from == edge.to {
            return;
        }
        self.heard.insert(
            (edge.from, edge.to),
            Heard {
                edge,
                received_ns: now_ns,
            },
        );
    }

    /// Forget heard edges received more than `max_age_ns` before `now_ns`. (Anchors age by
    /// [`expire_anchors`](Self::expire_anchors).)
    pub fn expire(&mut self, now_ns: u64, max_age_ns: u64) {
        self.heard
            .retain(|_, h| now_ns.saturating_sub(h.received_ns) <= max_age_ns);
    }

    /// Every edge heard from others.
    pub fn heard(&self) -> impl Iterator<Item = &EdgeEstimate> + '_ {
        self.heard.values().map(|h| &h.edge)
    }

    /// The raw measurements, oriented along each edge key.
    pub fn lut(&self) -> &TwistLUT {
        &self.lut
    }

    /// The network frame from everything this node knows, with `reference` held at zero and
    /// offsets read at `at_ns` on this node's clock.
    pub fn solve(&self, reference: NodeId, at_ns: u64) -> NetworkFrame {
        let mut edges = self.edges();
        edges.extend(self.heard().copied());
        solve_frame(self.me, reference, at_ns, &edges)
    }

    /// The absolute frame: every clock against [`UTC`], which is the consensus of the anchors
    /// weighted by their spreads, at `at_ns` on this node's clock. `None` without an anchor.
    ///
    /// Clock selection, as NTP does it: while at least three anchors remain and the worst one's
    /// edge to UTC sits more than five of its own spreads from the consensus, it is set aside as a
    /// falseticker and the frame solved again. With two anchors a disagreement shows as their
    /// cycle through UTC, but neither can be named the liar.
    ///
    /// The anchors' edges are read at `at_ns`, so node timescales should already be near UTC (one
    /// started from the system clock is); a timescale days from UTC would add its distance times
    /// the anchors' rate spread to theirs.
    pub fn solve_absolute(&self, at_ns: u64) -> Option<NetworkFrame> {
        let mut edges = self.edges();
        edges.extend(self.heard().copied());
        // Only anchors some measurement ties to the network: one nobody has an edge to says
        // nothing about any clock here, and would only sit in the frame looking agreed.
        let measured: BTreeSet<NodeId> = edges.iter().flat_map(|e| [e.from, e.to]).collect();
        let mut chosen: Vec<Anchor> = self
            .anchors()
            .into_iter()
            .filter(|a| measured.contains(&a.node))
            .collect();
        if chosen.is_empty() {
            return None;
        }
        let mut falsetickers = Vec::new();
        loop {
            let mut all = edges.clone();
            all.extend(chosen.iter().map(|a| EdgeEstimate {
                from: UTC,
                to: a.node,
                relation: Relation {
                    at_ns,
                    offset_ns: 0.0,
                    drift: 0.0,
                    sigma_offset_ns: a.sigma_offset_ns.max(MIN_SIGMA_NS),
                    sigma_drift: a.sigma_drift.max(MIN_SIGMA_DRIFT),
                },
                rtt_floor_ns: 0,
                samples: 0,
            }));
            let mut frame = solve_frame(self.me, UTC, at_ns, &all);
            let worst = frame
                .edges
                .iter()
                .filter(|e| e.from == UTC)
                .max_by(|a, b| a.z_offset.abs().total_cmp(&b.z_offset.abs()))
                .copied();
            match worst {
                Some(w) if chosen.len() >= 3 && w.z_offset.abs() > 5.0 => {
                    chosen.retain(|a| a.node != w.to);
                    falsetickers.push(w.to);
                }
                _ => {
                    frame.falsetickers = falsetickers;
                    return Some(frame);
                }
            }
        }
    }
}

/// One node in a solved frame, relative to the reference.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct NodeFrame {
    /// The node.
    pub id: NodeId,
    /// Its clock minus the reference's at the frame's instant, ns.
    pub offset_ns: f64,
    /// Its rate over the reference's, minus one.
    pub drift: f64,
    /// Standard deviation of `offset_ns`.
    pub sigma_offset_ns: f64,
    /// Standard deviation of `drift`.
    pub sigma_drift: f64,
}

impl NodeFrame {
    /// RTFSN's integer node state: offset in ns, drift in ppb, and a confidence of
    /// `1e6 / σ_offset` (1000 for a microsecond).
    pub fn chiral(&self) -> ChiralFrame {
        let confidence = (1e6 / self.sigma_offset_ns.max(1.0)).min(u32::MAX as f64) as u32;
        ChiralFrame::new(
            self.offset_ns.round() as i64,
            (self.drift * 1e9).round() as i64,
            confidence,
        )
    }
}

/// How far one edge sits from the solved frame.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EdgeResidual {
    /// Measuring node.
    pub from: NodeId,
    /// Measured node.
    pub to: NodeId,
    /// Measured minus solved offset at the frame's instant, ns.
    pub offset_ns: f64,
    /// Measured minus solved relative frequency.
    pub drift: f64,
    /// `offset_ns` over the edge's own standard deviation.
    pub z_offset: f64,
    /// `drift` over the edge's own standard deviation.
    pub z_drift: f64,
}

/// The holonomy of one fundamental cycle of the measured graph.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CycleDefect {
    /// The cycle, as walked (the first node is not repeated at the end).
    pub nodes: Vec<NodeId>,
    /// Sum of the offsets around the cycle, ns. Zero for a consistent clock geometry.
    pub offset_ns: f64,
    /// Sum of the log relative frequencies around the cycle.
    pub drift: f64,
    /// The spread the edges' own uncertainties predict for `offset_ns`.
    pub sigma_offset_ns: f64,
    /// The spread predicted for `drift`.
    pub sigma_drift: f64,
    /// The larger of the two normalized defects.
    pub z: f64,
}

/// A solved network frame.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NetworkFrame {
    /// The node held at zero.
    pub reference: NodeId,
    /// The node whose clock `at_ns` is read on.
    pub solver: NodeId,
    /// The instant of the offsets, on the solver's clock.
    pub at_ns: u64,
    /// Whether the solver is connected to the reference, so that `at_ns` could be carried onto
    /// every clock. When it is not, each edge is read at its own instant.
    pub instant_mapped: bool,
    /// Every node connected to the reference, the reference included.
    pub nodes: Vec<NodeFrame>,
    /// Nodes measured but not connected to the reference.
    pub unreachable: Vec<NodeId>,
    /// Every edge used, with its residual.
    pub edges: Vec<EdgeResidual>,
    /// The fundamental cycles' holonomy, worst first.
    pub cycles: Vec<CycleDefect>,
    /// Root mean square of the edges' offset residuals, ns.
    pub rms_offset_ns: f64,
    /// Root mean square of the edges' relative-frequency residuals.
    pub rms_drift: f64,
    /// Anchors set aside by clock selection in an absolute frame.
    #[serde(default)]
    pub falsetickers: Vec<NodeId>,
}

impl NetworkFrame {
    /// The frame of one node.
    pub fn node(&self, id: NodeId) -> Option<&NodeFrame> {
        self.nodes.iter().find(|n| n.id == id)
    }

    /// Whether this frame is against UTC (solved by [`RelationalClock::solve_absolute`]).
    pub fn is_absolute(&self) -> bool {
        self.reference == UTC
    }
}

const MIN_SIGMA_NS: f64 = 1.0;
const MIN_SIGMA_DRIFT: f64 = 1e-13;

/// Solve `A x = b` for symmetric positive-definite `A` (row-major, `n × n`) by Cholesky, with the
/// diagonal of `A⁻¹`. `None` when `A` is not positive definite.
fn spd_solve(a: &[f64], b: &[f64], n: usize) -> Option<(Vec<f64>, Vec<f64>)> {
    let mut l = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                if s <= 0.0 || !s.is_finite() {
                    return None;
                }
                l[i * n + i] = s.sqrt();
            } else {
                l[i * n + j] = s / l[j * n + j];
            }
        }
    }
    let solve = |rhs: &[f64]| -> Vec<f64> {
        let mut y = vec![0.0; n];
        for i in 0..n {
            let mut s = rhs[i];
            for k in 0..i {
                s -= l[i * n + k] * y[k];
            }
            y[i] = s / l[i * n + i];
        }
        let mut x = vec![0.0; n];
        for i in (0..n).rev() {
            let mut s = y[i];
            for k in i + 1..n {
                s -= l[k * n + i] * x[k];
            }
            x[i] = s / l[i * n + i];
        }
        x
    };
    let x = solve(b);
    let mut diag = vec![0.0; n];
    let mut e = vec![0.0; n];
    for i in 0..n {
        e[i] = 1.0;
        diag[i] = solve(&e)[i];
        e[i] = 0.0;
    }
    Some((x, diag))
}

/// One weighted measurement of `value[b] − value[a]`.
struct Obs {
    a: usize,
    b: usize,
    m: f64,
    sigma: f64,
}

/// Weighted least squares for node values from differences, with node `fixed` held at zero.
/// Returns each node's value and standard deviation; nodes not in `members` stay zero.
fn laplacian_solve(
    n: usize,
    fixed: usize,
    members: &BTreeSet<usize>,
    obs: &[Obs],
) -> (Vec<f64>, Vec<f64>) {
    let unknowns: Vec<usize> = members.iter().copied().filter(|&i| i != fixed).collect();
    let pos: BTreeMap<usize, usize> = unknowns.iter().enumerate().map(|(p, &i)| (i, p)).collect();
    let k = unknowns.len();
    let mut mat = vec![0.0; k * k];
    let mut rhs = vec![0.0; k];
    for o in obs {
        let w = 1.0 / o.sigma.powi(2);
        let (pa, pb) = (pos.get(&o.a).copied(), pos.get(&o.b).copied());
        if let Some(pa) = pa {
            mat[pa * k + pa] += w;
            rhs[pa] -= w * o.m;
        }
        if let Some(pb) = pb {
            mat[pb * k + pb] += w;
            rhs[pb] += w * o.m;
        }
        if let (Some(pa), Some(pb)) = (pa, pb) {
            mat[pa * k + pb] -= w;
            mat[pb * k + pa] -= w;
        }
    }
    let mut value = vec![0.0; n];
    let mut sigma = vec![0.0; n];
    if let Some((x, diag)) = spd_solve(&mat, &rhs, k) {
        for (p, &i) in unknowns.iter().enumerate() {
            value[i] = x[p];
            sigma[i] = diag[p].max(0.0).sqrt();
        }
    }
    (value, sigma)
}

/// The network frame from a set of directed edge estimates. `solver` is the node whose clock
/// `at_ns` is read on; `reference` is held at zero.
pub fn solve_frame(
    solver: NodeId,
    reference: NodeId,
    at_ns: u64,
    edges: &[EdgeEstimate],
) -> NetworkFrame {
    let mut ids: BTreeSet<NodeId> = BTreeSet::from([solver, reference]);
    for e in edges {
        ids.insert(e.from);
        ids.insert(e.to);
    }
    let ids: Vec<NodeId> = ids.into_iter().collect();
    let index = |id: NodeId| ids.binary_search(&id).expect("every endpoint is listed");
    let n = ids.len();
    let usable: Vec<&EdgeEstimate> = edges
        .iter()
        .filter(|e| {
            e.from != e.to
                && e.relation.offset_ns.is_finite()
                && e.relation.drift.is_finite()
                && e.relation.drift > -1.0
        })
        .collect();

    // The component of the reference.
    let mut adj: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); n];
    for e in &usable {
        let (a, b) = (index(e.from), index(e.to));
        adj[a].insert(b);
        adj[b].insert(a);
    }
    let r = index(reference);
    let mut members = BTreeSet::from([r]);
    let mut queue = VecDeque::from([r]);
    let mut parent: Vec<Option<usize>> = vec![None; n];
    let mut order = Vec::new();
    while let Some(u) = queue.pop_front() {
        order.push(u);
        for &v in &adj[u] {
            if members.insert(v) {
                parent[v] = Some(u);
                queue.push_back(v);
            }
        }
    }
    let me = index(solver);
    let instant_mapped = members.contains(&me);
    let inside: Vec<&EdgeEstimate> = usable
        .into_iter()
        .filter(|e| members.contains(&index(e.from)))
        .collect();

    // Relative frequencies, in logs so that composition is exact: ln(1+y_b) − ln(1+y_a) = ln(1+d).
    let drift_obs: Vec<Obs> = inside
        .iter()
        .map(|e| Obs {
            a: index(e.from),
            b: index(e.to),
            m: e.relation.drift.ln_1p(),
            sigma: (e.relation.sigma_drift / (1.0 + e.relation.drift)).max(MIN_SIGMA_DRIFT),
        })
        .collect();
    let (log_rate, sigma_log_rate) = laplacian_solve(n, r, &members, &drift_obs);

    // Offsets at one instant. Each edge is read at its measuring node's clock at that instant,
    // which needs the offsets themselves; three passes settle it to well under a nanosecond.
    let mut offset = vec![0.0; n];
    let mut sigma_offset = vec![0.0; n];
    let mut offset_obs: Vec<Obs> = Vec::new();
    for _ in 0..3 {
        offset_obs = inside
            .iter()
            .map(|e| {
                let a = index(e.from);
                let when = if instant_mapped {
                    at_ns as f64 + offset[a] - offset[me]
                } else {
                    e.relation.at_ns as f64
                };
                Obs {
                    a,
                    b: index(e.to),
                    m: e.relation.offset_at(when),
                    sigma: e.relation.sigma_at(when).max(MIN_SIGMA_NS),
                }
            })
            .collect();
        let (o, s) = laplacian_solve(n, r, &members, &offset_obs);
        offset = o;
        sigma_offset = s;
    }

    let mut residuals: Vec<EdgeResidual> = inside
        .iter()
        .zip(offset_obs.iter().zip(drift_obs.iter()))
        .map(|(e, (oo, od))| {
            let ro = oo.m - (offset[oo.b] - offset[oo.a]);
            let rd = od.m - (log_rate[od.b] - log_rate[od.a]);
            EdgeResidual {
                from: e.from,
                to: e.to,
                offset_ns: ro,
                drift: rd,
                z_offset: ro / oo.sigma,
                z_drift: rd / od.sigma,
            }
        })
        .collect();
    let count = residuals.len().max(1) as f64;
    let rms_offset_ns = (residuals.iter().map(|r| r.offset_ns.powi(2)).sum::<f64>() / count).sqrt();
    let rms_drift = (residuals.iter().map(|r| r.drift.powi(2)).sum::<f64>() / count).sqrt();
    residuals.sort_by(|a, b| {
        let za = a.z_offset.abs().max(a.z_drift.abs());
        let zb = b.z_offset.abs().max(b.z_drift.abs());
        zb.total_cmp(&za)
    });

    // Cycles: merge each pair's measurements (both directions, inverse-variance weighted) into
    // one oriented low → high, then walk the BFS tree for every pair that is not a tree edge.
    struct Pair {
        o_sum: f64,
        o_w: f64,
        d_sum: f64,
        d_w: f64,
    }
    let mut pairs: BTreeMap<(usize, usize), Pair> = BTreeMap::new();
    for (oo, od) in offset_obs.iter().zip(drift_obs.iter()) {
        let (lo, hi, sign) = if oo.a < oo.b {
            (oo.a, oo.b, 1.0)
        } else {
            (oo.b, oo.a, -1.0)
        };
        let p = pairs.entry((lo, hi)).or_insert(Pair {
            o_sum: 0.0,
            o_w: 0.0,
            d_sum: 0.0,
            d_w: 0.0,
        });
        let (wo, wd) = (1.0 / oo.sigma.powi(2), 1.0 / od.sigma.powi(2));
        p.o_sum += sign * oo.m * wo;
        p.o_w += wo;
        p.d_sum += sign * od.m * wd;
        p.d_w += wd;
    }
    // (offset, σ², drift, σ²) of walking a → b.
    let walk = |a: usize, b: usize| -> (f64, f64, f64, f64) {
        let (lo, hi, sign) = if a < b { (a, b, 1.0) } else { (b, a, -1.0) };
        let p = &pairs[&(lo, hi)];
        (
            sign * p.o_sum / p.o_w,
            1.0 / p.o_w,
            sign * p.d_sum / p.d_w,
            1.0 / p.d_w,
        )
    };
    let depth = {
        let mut d = vec![0usize; n];
        for &u in &order {
            if let Some(p) = parent[u] {
                d[u] = d[p] + 1;
            }
        }
        d
    };
    let mut cycles = Vec::new();
    for &(u, v) in pairs.keys() {
        if parent[v] == Some(u) || parent[u] == Some(v) {
            continue;
        }
        // Tree paths from u and v up to their lowest common ancestor.
        let (mut x, mut y) = (u, v);
        let (mut up_u, mut up_v) = (vec![u], vec![v]);
        while x != y {
            if depth[x] >= depth[y] {
                x = parent[x].expect("non-root has a parent");
                up_u.push(x);
            } else {
                y = parent[y].expect("non-root has a parent");
                up_v.push(y);
            }
        }
        // Walk u → v on the pair, then v up to the ancestor and down to u.
        let mut walk_nodes = vec![u, v];
        walk_nodes.extend(up_v.iter().skip(1));
        walk_nodes.extend(up_u.iter().rev().skip(1).take(up_u.len().saturating_sub(2)));
        let (mut o, mut so, mut d, mut sd) = (0.0, 0.0, 0.0, 0.0);
        for i in 0..walk_nodes.len() {
            let (a, b) = (walk_nodes[i], walk_nodes[(i + 1) % walk_nodes.len()]);
            let (wo, wso, wd, wsd) = walk(a, b);
            o += wo;
            so += wso;
            d += wd;
            sd += wsd;
        }
        let (so, sd) = (so.sqrt(), sd.sqrt());
        cycles.push(CycleDefect {
            nodes: walk_nodes.iter().map(|&i| ids[i]).collect(),
            offset_ns: o,
            drift: d,
            sigma_offset_ns: so,
            sigma_drift: sd,
            z: (o / so).abs().max((d / sd).abs()),
        });
    }
    cycles.sort_by(|a, b| b.z.total_cmp(&a.z));

    let nodes = order
        .iter()
        .map(|&i| NodeFrame {
            id: ids[i],
            offset_ns: offset[i],
            drift: log_rate[i].exp_m1(),
            sigma_offset_ns: sigma_offset[i],
            sigma_drift: sigma_log_rate[i] * log_rate[i].exp(),
        })
        .collect();
    let unreachable = (0..n)
        .filter(|i| !members.contains(i))
        .map(|i| ids[i])
        .collect();
    NetworkFrame {
        reference,
        solver,
        at_ns,
        instant_mapped,
        nodes,
        unreachable,
        edges: residuals,
        cycles,
        rms_offset_ns,
        rms_drift,
        falsetickers: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    /// A free-running oscillator: reads `offset + t · (1 + ppm·1e-6)` at true time `t` (ns).
    #[derive(Clone, Copy)]
    struct Osc {
        offset_ns: f64,
        ppm: f64,
    }

    impl Osc {
        fn read(&self, t: f64) -> f64 {
            self.offset_ns + t * (1.0 + self.ppm * 1e-6)
        }
        /// True time at which this oscillator reads `local`.
        fn inverse(&self, local: f64) -> f64 {
            (local - self.offset_ns) / (1.0 + self.ppm * 1e-6)
        }
    }

    fn id(i: u8) -> NodeId {
        let mut b = [0u8; 32];
        b[0] = i;
        NodeId(b)
    }

    /// The link between two nodes: a fixed delay each way, uniform queueing, rare spikes.
    #[derive(Clone, Copy)]
    struct Link {
        base_ab_ns: f64,
        base_ba_ns: f64,
        jitter_ns: f64,
        spike_p: f64,
    }

    const LAN: Link = Link {
        base_ab_ns: 150_000.0,
        base_ba_ns: 150_000.0,
        jitter_ns: 100_000.0,
        spike_p: 0.05,
    };

    /// One exchange initiated by `a` at true time `t`; `lie_ns` is added to every timestamp the
    /// responder `b` reports.
    fn exchange(
        rng: &mut StdRng,
        a: Osc,
        b: Osc,
        link: Link,
        t: f64,
        lie_ns: f64,
    ) -> (u64, u64, u64, u64) {
        let mut leg = |base: f64| {
            let spike = if rng.random::<f64>() < link.spike_p {
                2_000_000.0
            } else {
                0.0
            };
            base + rng.random::<f64>() * link.jitter_ns + spike
        };
        let fwd = leg(link.base_ab_ns);
        let back = leg(link.base_ba_ns);
        let mut stamp = || (rng.random::<f64>() - 0.5) * 20_000.0;
        let t1 = a.read(t);
        let t2 = b.read(t + fwd) + stamp() + lie_ns;
        let t3 = b.read(t + fwd + 40_000.0) + stamp() + lie_ns;
        let t4 = a.read(t + fwd + 40_000.0 + back) + stamp();
        (t1 as u64, t2 as u64, t3 as u64, t4 as u64)
    }

    #[test]
    fn spd_solve_matches_a_known_system() {
        // [[4, 2], [2, 3]] x = [2, 1] → x = [0.5, 0]; inverse diag = [3/8, 1/2].
        let (x, d) = spd_solve(&[4.0, 2.0, 2.0, 3.0], &[2.0, 1.0], 2).unwrap();
        assert!((x[0] - 0.5).abs() < 1e-12 && x[1].abs() < 1e-12);
        assert!((d[0] - 0.375).abs() < 1e-12 && (d[1] - 0.5).abs() < 1e-12);
        assert!(spd_solve(&[1.0, 2.0, 2.0, 1.0], &[0.0, 0.0], 2).is_none());
    }

    #[test]
    fn a_track_recovers_offset_and_relative_frequency() {
        let mut rng = StdRng::seed_from_u64(7);
        let a = Osc {
            offset_ns: 0.0,
            ppm: 0.0,
        };
        let b = Osc {
            offset_ns: 2.5e9,
            ppm: 30.0,
        };
        let mut track = PeerTrack::new(TrackConfig::default());
        let (mut accepted, mut popcorn) = (0, 0);
        for k in 0..480 {
            let t = 1e9 + k as f64 * 250e6;
            let (t1, t2, t3, t4) = exchange(&mut rng, a, b, LAN, t, 0.0);
            match track.observe(t1, t2, t3, t4) {
                Observation::Accepted { .. } => accepted += 1,
                Observation::Rejected {
                    reason: Reject::Popcorn { .. },
                } => popcorn += 1,
                other => panic!("unexpected {other:?}"),
            }
        }
        assert!(
            popcorn > 0 && accepted > 300,
            "accepted {accepted}, popcorn {popcorn}"
        );
        let rel = track.relation().unwrap();
        let truth_t = a.inverse(rel.at_ns as f64);
        let true_offset = b.read(truth_t) - a.read(truth_t);
        let true_drift = (1.0 + 30e-6) / 1.0 - 1.0;
        let off_err = rel.offset_ns - true_offset;
        let drift_err = rel.drift - true_drift;
        assert!(off_err.abs() < 10_000.0, "offset error {off_err} ns");
        assert!(drift_err.abs() < 3e-7, "drift error {drift_err}");
        assert!(
            rel.sigma_offset_ns < 20_000.0 && rel.sigma_drift < 1e-6,
            "{rel:?}"
        );
    }

    const OSCS: [Osc; 5] = [
        Osc {
            offset_ns: 0.0,
            ppm: 0.0,
        },
        Osc {
            offset_ns: 1.2e9,
            ppm: 12.0,
        },
        Osc {
            offset_ns: -3.0e8,
            ppm: -7.5,
        },
        Osc {
            offset_ns: 5.5e9,
            ppm: 40.0,
        },
        Osc {
            offset_ns: 7.7e7,
            ppm: 0.5,
        },
    ];

    /// Every node exchanges with every other, both ways, every 500 ms for two minutes; then each
    /// node gossips its edges to the others. `link` gives the link for an ordered pair and `lie`
    /// what the responder adds to its timestamps.
    fn mesh(
        seed: u64,
        link: impl Fn(usize, usize) -> Link,
        lie: impl Fn(usize, usize) -> f64,
    ) -> (Vec<RelationalClock>, f64) {
        let mut rng = StdRng::seed_from_u64(seed);
        let n = OSCS.len();
        let mut clocks: Vec<RelationalClock> = (0..n)
            .map(|i| RelationalClock::new(id(i as u8), TrackConfig::default()))
            .collect();
        let mut t = 1e9;
        for _ in 0..240 {
            for a in 0..n {
                for b in 0..n {
                    if a != b {
                        t += 1e6;
                        let ab = link(a, b);
                        let l = Link {
                            base_ab_ns: ab.base_ab_ns,
                            base_ba_ns: link(b, a).base_ab_ns,
                            ..ab
                        };
                        let (t1, t2, t3, t4) =
                            exchange(&mut rng, OSCS[a], OSCS[b], l, t, lie(a, b));
                        clocks[a].observe(id(b as u8), t1, t2, t3, t4);
                    }
                }
            }
            t += 500e6 - (n * (n - 1)) as f64 * 1e6;
        }
        let all: Vec<Vec<EdgeEstimate>> = clocks.iter().map(|c| c.edges()).collect();
        for (i, c) in clocks.iter_mut().enumerate() {
            let now = OSCS[i].read(t) as u64;
            for (j, edges) in all.iter().enumerate() {
                if i != j {
                    for e in edges {
                        c.hear(*e, now);
                    }
                }
            }
        }
        (clocks, t)
    }

    #[test]
    fn a_mesh_solves_to_the_true_frame_from_any_vantage() {
        let (clocks, t) = mesh(11, |_, _| LAN, |_, _| 0.0);
        for solver in [0usize, 3] {
            let at = OSCS[solver].read(t) as u64;
            let frame = clocks[solver].solve(id(0), at);
            assert!(frame.instant_mapped && frame.unreachable.is_empty());
            let true_t = OSCS[solver].inverse(at as f64);
            for (i, osc) in OSCS.iter().enumerate() {
                let nf = frame.node(id(i as u8)).unwrap();
                let true_offset = osc.read(true_t) - OSCS[0].read(true_t);
                let true_drift = (1.0 + osc.ppm * 1e-6) / (1.0 + OSCS[0].ppm * 1e-6) - 1.0;
                assert!(
                    (nf.offset_ns - true_offset).abs() < 15_000.0,
                    "solver {solver}, node {i}: offset {} vs {true_offset}",
                    nf.offset_ns
                );
                assert!(
                    (nf.drift - true_drift).abs() < 3e-7,
                    "solver {solver}, node {i}: drift {} vs {true_drift}",
                    nf.drift
                );
            }
            // 5 nodes, 10 pairs: 10 − 4 = 6 independent cycles, all consistent.
            assert_eq!(frame.cycles.len(), 6);
            for c in &frame.cycles {
                assert!(c.z < 5.0 && c.offset_ns.abs() < 30_000.0, "{c:?}");
            }
        }
    }

    #[test]
    fn an_asymmetric_link_leaves_holonomy_on_its_cycles() {
        // 1 → 2 is 2 ms slower than 2 → 1: both ends read the pair 1 ms off.
        let slow = |a: usize, b: usize| {
            if (a, b) == (1, 2) {
                Link {
                    base_ab_ns: 2_150_000.0,
                    ..LAN
                }
            } else {
                LAN
            }
        };
        let (clocks, t) = mesh(13, slow, |_, _| 0.0);
        let frame = clocks[0].solve(id(0), OSCS[0].read(t) as u64);
        let worst = frame.edges[0];
        let pair = BTreeSet::from([worst.from, worst.to]);
        assert_eq!(pair, BTreeSet::from([id(1), id(2)]), "{worst:?}");
        let through: Vec<&CycleDefect> = frame
            .cycles
            .iter()
            .filter(|c| {
                c.nodes
                    .windows(2)
                    .chain(std::iter::once(&[*c.nodes.last().unwrap(), c.nodes[0]][..]))
                    .any(|w| BTreeSet::from([w[0], w[1]]) == BTreeSet::from([id(1), id(2)]))
            })
            .collect();
        assert!(!through.is_empty());
        for c in &frame.cycles {
            let crosses = through.iter().any(|x| x.nodes == c.nodes);
            if crosses {
                assert!(
                    (c.offset_ns.abs() - 1_000_000.0).abs() < 100_000.0 && c.z > 10.0,
                    "{c:?}"
                );
            } else {
                assert!(c.z < 5.0, "{c:?}");
            }
        }
    }

    #[test]
    fn a_node_that_lies_to_one_peer_is_seen_and_a_consistent_liar_is_not() {
        // Node 3 adds 5 ms to what it tells node 1 only: inconsistent, so it leaves holonomy.
        let (clocks, t) = mesh(
            17,
            |_, _| LAN,
            |a, b| if (a, b) == (1, 3) { 5e6 } else { 0.0 },
        );
        let frame = clocks[0].solve(id(0), OSCS[0].read(t) as u64);
        assert!(frame.cycles[0].z > 10.0, "{:?}", frame.cycles[0]);

        // Node 3 adds 5 ms to everything it reports and to its own clock when it initiates:
        // a coboundary. No cycle moves; the frame simply puts node 3 5 ms later.
        let liar = Osc {
            offset_ns: OSCS[3].offset_ns + 5e6,
            ..OSCS[3]
        };
        let mut rng = StdRng::seed_from_u64(19);
        let mut clocks: Vec<RelationalClock> = (0..5)
            .map(|i| RelationalClock::new(id(i as u8), TrackConfig::default()))
            .collect();
        let osc = |i: usize| if i == 3 { liar } else { OSCS[i] };
        let mut t = 1e9;
        for _ in 0..240 {
            for a in 0..5 {
                for b in 0..5 {
                    if a != b {
                        t += 1e6;
                        let (t1, t2, t3, t4) = exchange(&mut rng, osc(a), osc(b), LAN, t, 0.0);
                        clocks[a].observe(id(b as u8), t1, t2, t3, t4);
                    }
                }
            }
            t += 480e6;
        }
        let all: Vec<EdgeEstimate> = clocks[1..].iter().flat_map(|c| c.edges()).collect();
        for e in all {
            clocks[0].hear(e, 0);
        }
        let at = OSCS[0].read(t) as u64;
        let frame = clocks[0].solve(id(0), at);
        for c in &frame.cycles {
            assert!(c.z < 5.0, "a coboundary leaves no holonomy: {c:?}");
        }
        let true_t = OSCS[0].inverse(at as f64);
        let honest = OSCS[3].read(true_t) - OSCS[0].read(true_t);
        let solved = frame.node(id(3)).unwrap().offset_ns;
        assert!(
            (solved - honest - 5e6).abs() < 20_000.0,
            "{solved} vs {honest}"
        );
    }

    #[test]
    fn a_disconnected_node_is_unreachable_and_heard_edges_expire() {
        let mut c = RelationalClock::new(id(0), TrackConfig::default());
        let rel = Relation {
            at_ns: 0,
            offset_ns: 1e6,
            drift: 0.0,
            sigma_offset_ns: 1e3,
            sigma_drift: 1e-9,
        };
        let edge = |a: u8, b: u8| EdgeEstimate {
            from: id(a),
            to: id(b),
            relation: rel,
            rtt_floor_ns: 1,
            samples: 1,
        };
        c.hear(edge(1, 2), 100);
        c.hear(edge(0, 5), 100); // ours to measure: ignored
        let frame = c.solve(id(0), 0);
        assert!(!frame.instant_mapped || frame.nodes.len() == 1);
        assert_eq!(frame.unreachable, vec![id(1), id(2)]);
        c.hear(edge(2, 3), 900);
        c.expire(1_000, 500);
        assert_eq!(c.heard().count(), 1);
        let chiral = NodeFrame {
            id: id(9),
            offset_ns: 1500.4,
            drift: 2.5e-6,
            sigma_offset_ns: 1000.0,
            sigma_drift: 0.0,
        }
        .chiral();
        assert_eq!(
            (chiral.offset_nanos, chiral.drift_ppb, chiral.confidence),
            (1500, 2500, 1000)
        );
    }
    #[test]
    fn a_peer_whose_timescale_steps_is_tracked_afresh() {
        let mut rng = StdRng::seed_from_u64(23);
        let a = Osc {
            offset_ns: 0.0,
            ppm: 0.0,
        };
        let b = Osc {
            offset_ns: 4e6,
            ppm: 17.0,
        };
        let restarted = Osc {
            offset_ns: 2.004e9,
            ppm: 17.0,
        };
        let mut track = PeerTrack::new(TrackConfig::default());
        let mut t = 1e9;
        for _ in 0..200 {
            let (t1, t2, t3, t4) = exchange(&mut rng, a, b, LAN, t, 0.0);
            track.observe(t1, t2, t3, t4);
            t += 250e6;
        }
        let mut outcomes = Vec::new();
        for _ in 0..3 {
            // Quiet exchanges, so only the step itself can trip the check.
            let quiet = Link {
                jitter_ns: 0.0,
                spike_p: 0.0,
                ..LAN
            };
            let (t1, t2, t3, t4) = exchange(&mut rng, a, restarted, quiet, t, 0.0);
            outcomes.push(track.observe(t1, t2, t3, t4));
            t += 250e6;
        }
        assert!(
            matches!(
                outcomes[0],
                Observation::Rejected { reason: Reject::Step { jump_ns } } if (jump_ns - 2_000_000_000).abs() < 1_000_000
            ),
            "{:?}",
            outcomes[0]
        );
        assert!(
            matches!(
                outcomes[1],
                Observation::Accepted {
                    restarted: true,
                    ..
                }
            ),
            "{:?}",
            outcomes[1]
        );
        assert!(
            matches!(
                outcomes[2],
                Observation::Accepted {
                    restarted: false,
                    ..
                }
            ),
            "{:?}",
            outcomes[2]
        );
        assert_eq!(track.steps, 1);
        let rel = track.relation().unwrap();
        let truth = restarted.read(a.inverse(rel.at_ns as f64)) - rel.at_ns as f64;
        assert!(
            (rel.offset_ns - truth).abs() < 50_000.0,
            "{} vs {truth}",
            rel.offset_ns
        );
    }

    /// Clocks near UTC (milliseconds off, tens of ppm), meshed, and NTP servers: `A` on the LAN,
    /// 200 µs fast with a 50 µs root distance; `B` across a WAN, 300 µs slow with 2 ms; and `C`,
    /// 40 ms fast while claiming 100 µs. Returns node 0 after hearing everyone, and the time.
    fn anchored(servers: &[(u8, f64, f64, Link)]) -> (RelationalClock, f64) {
        const NODES: [Osc; 4] = [
            Osc {
                offset_ns: 1.2e6,
                ppm: 0.0,
            },
            Osc {
                offset_ns: -0.8e6,
                ppm: 12.0,
            },
            Osc {
                offset_ns: 3.1e6,
                ppm: -7.5,
            },
            Osc {
                offset_ns: -2.0e6,
                ppm: 40.0,
            },
        ];
        let mut rng = StdRng::seed_from_u64(29);
        let mut clocks: Vec<RelationalClock> = (0..4)
            .map(|i| RelationalClock::new(id(i as u8), TrackConfig::default()))
            .collect();
        let mut t = 1e9;
        for _ in 0..240 {
            for a in 0..4 {
                for b in 0..4 {
                    if a != b {
                        let (t1, t2, t3, t4) = exchange(&mut rng, NODES[a], NODES[b], LAN, t, 0.0);
                        clocks[a].observe(id(b as u8), t1, t2, t3, t4);
                        t += 1e6;
                    }
                }
                // Each server is asked by one node: server k by node k.
                if let Some(&(sid, bias, root, link)) = servers.get(a) {
                    let server = Osc {
                        offset_ns: bias,
                        ppm: 0.0,
                    };
                    let ex = exchange(&mut rng, NODES[a], server, link, t, 0.0);
                    let now = NODES[a].read(t) as u64;
                    clocks[a].observe_reference(id(sid), ex, root, 2, 5e-8, now);
                    t += 1e6;
                }
            }
            t += 400e6;
        }
        let edges: Vec<EdgeEstimate> = clocks[1..].iter().flat_map(|c| c.edges()).collect();
        let anchors: Vec<Anchor> = clocks[1..].iter().flat_map(|c| c.own_anchors()).collect();
        let now = NODES[0].read(t) as u64;
        let mut c0 = clocks.swap_remove(0);
        for e in edges {
            c0.hear(e, now);
        }
        for a in anchors {
            c0.hear_anchor(a, now);
        }
        // Check each node against truth here, where the oscillators are known.
        let frame = c0.solve_absolute(now).expect("anchored");
        let true_t = NODES[0].inverse(now as f64);
        let chosen: Vec<&(u8, f64, f64, Link)> = servers
            .iter()
            .filter(|s| !frame.falsetickers.contains(&id(s.0)))
            .collect();
        let (mut wsum, mut bsum) = (0.0, 0.0);
        for (sid, bias, _, _) in &chosen {
            let a = frame_anchor(&c0, id(*sid));
            let w = 1.0 / a.sigma_offset_ns.powi(2);
            wsum += w;
            bsum += w * bias;
        }
        let consensus = bsum / wsum;
        for (i, osc) in NODES.iter().enumerate() {
            let nf = frame.node(id(i as u8)).unwrap();
            let truth = osc.read(true_t) - true_t;
            assert!(
                (nf.offset_ns - (truth - consensus)).abs() < 40_000.0,
                "node {i}: {} vs {} (truth {truth}, consensus bias {consensus})",
                nf.offset_ns,
                truth - consensus
            );
            assert!(
                (nf.drift * 1e6 - osc.ppm).abs() < 0.3,
                "node {i}: {} ppm",
                nf.drift * 1e6
            );
        }
        (c0, t)
    }

    /// A re-gossiped anchor keeps the age of its last measurement, so one nobody queries any
    /// more expires even while nodes keep handing it to each other; and an anchor no edge reaches
    /// stays out of the frame.
    #[test]
    fn a_stale_anchor_ages_out_and_an_unmeasured_one_stays_out() {
        let s = 1_000_000_000u64;
        let anchor = |n: u8, age_ns: u64| Anchor {
            node: id(n),
            sigma_offset_ns: 1e6,
            sigma_drift: 5e-8,
            stratum: 2,
            age_ns,
        };
        let mut a = RelationalClock::new(id(0), TrackConfig::default());
        let mut b = RelationalClock::new(id(1), TrackConfig::default());
        // Node 0 heard of server 9, measured by someone 5 s before.
        a.hear_anchor(anchor(9, 5 * s), 100 * s);
        // Nodes 0 and 1 hand it back and forth every second for 20 minutes; nobody measures it.
        let mut t = 100 * s;
        for _ in 0..1200 {
            t += s;
            for x in a.gossip_anchors(t) {
                b.hear_anchor(x, t);
            }
            for x in b.gossip_anchors(t) {
                a.hear_anchor(x, t);
            }
            a.expire_anchors(t, 900 * s);
            b.expire_anchors(t, 900 * s);
        }
        assert!(a.anchors().is_empty() && b.anchors().is_empty(), "the anchor circulated");
        // An anchor with no edge to it is not used: no absolute frame from it alone.
        a.hear_anchor(anchor(8, 0), t);
        assert_eq!(a.anchors().len(), 1);
        assert!(a.solve_absolute(t).is_none());
    }

    fn frame_anchor(c: &RelationalClock, node: NodeId) -> Anchor {
        c.anchors().into_iter().find(|a| a.node == node).unwrap()
    }

    const WAN: Link = Link {
        base_ab_ns: 15e6,
        base_ba_ns: 15e6,
        jitter_ns: 400_000.0,
        spike_p: 0.05,
    };

    #[test]
    fn anchors_put_every_clock_on_utc_and_a_falseticker_is_set_aside() {
        let (c0, t) = anchored(&[
            (200, 200_000.0, 50_000.0, LAN),
            (201, -300_000.0, 2e6, WAN),
            (202, 40e6, 100_000.0, LAN),
        ]);
        let frame = c0.solve_absolute(t as u64).unwrap();
        assert!(frame.is_absolute());
        assert_eq!(frame.falsetickers, vec![id(202)]);
        let a = frame_anchor(&c0, id(200));
        assert!(
            a.sigma_offset_ns > 50_000.0 && a.sigma_offset_ns < 300_000.0,
            "{a:?}"
        );
    }

    #[test]
    fn two_disagreeing_anchors_show_as_a_cycle_through_utc() {
        // Two that agree within their spreads: no falseticker, every clock on UTC (checked inside).
        let (c0, t) = anchored(&[(200, 200_000.0, 50_000.0, LAN), (201, -300_000.0, 2e6, WAN)]);
        assert!(c0.solve_absolute(t as u64).unwrap().falsetickers.is_empty());
        // Two 40 ms apart, each claiming 100 µs: no majority to name the liar, but their
        // disagreement is a cycle through UTC that cannot close.
        let mut rng = StdRng::seed_from_u64(37);
        let node = Osc {
            offset_ns: 1e6,
            ppm: 5.0,
        };
        let mut c = RelationalClock::new(id(0), TrackConfig::default());
        let mut t = 1e9;
        for _ in 0..120 {
            for (sid, bias) in [(200u8, 200_000.0), (201u8, 40e6)] {
                let ex = exchange(
                    &mut rng,
                    node,
                    Osc {
                        offset_ns: bias,
                        ppm: 0.0,
                    },
                    LAN,
                    t,
                    0.0,
                );
                c.observe_reference(id(sid), ex, 100_000.0, 1, 5e-8, node.read(t) as u64);
                t += 1e6;
            }
            t += 500e6;
        }
        let frame = c.solve_absolute(node.read(t) as u64).unwrap();
        assert!(frame.falsetickers.is_empty());
        let through_utc = frame
            .cycles
            .iter()
            .find(|c| c.nodes.contains(&UTC))
            .expect("the two anchors close a cycle through UTC");
        assert!(
            through_utc.z > 10.0 && (through_utc.offset_ns.abs() - 39.8e6).abs() < 1e6,
            "{through_utc:?}"
        );
    }
}
