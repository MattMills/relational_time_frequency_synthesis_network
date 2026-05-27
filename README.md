# Relational Time-Frequency Synthesis Network (RTFSN)

A distributed time synchronization protocol built on nested DHT layers with
cryptographic blinding, implementing a phase manifold coherence solver that
produces a privacy-preserving consensus clock.

## Architecture

Four concentric DHT layers, each with independent key spaces, routing tables,
and consistency guarantees. Information flows inward freely but passes through
irreversible cryptographic valves when flowing outward.

```
Layer 0 (Core)    ─ Identity & raw measurements      [private, encrypted]
    ↓ Pedersen commitment + ZK proof of identity
Layer 1           ─ Geometric embedding               [pseudonymous per epoch]
    ↓ Homomorphic aggregation into cluster commitments
Layer 2           ─ Consensus clock                    [cluster-level commitments]
    ↓ Threshold decryption + VDF anchor
Layer 3 (Surface) ─ Anonymous clock reference stream   [single scalar per epoch]
```

The only thing visible from outside is a monotonically increasing time value
backed by the distributed consensus, revealing nothing about who produced it.

## Modules

### `rtfsn-core`
Core protocol types, cryptographic primitives, and algorithms:

- **crypto/** — Pedersen commitments, identity blinding (Schnorr proofs),
  Ed25519 signatures, verifiable delay functions
- **holonomy/** — Phase manifold solver mapping clock sync to holonomy
  consistency: `ChiralFrame` (integer-basis node state), `TwistLUT` (pairwise
  measurements), `HolonomySolver` (iterative defect minimization), `SAMRState`
  (prime-channel adaptive scheduling), `TemporalMirror` (retrodictive-predictive
  fixed-point)
- **sync/** — NTP-like clock offset/drift estimation, Vivaldi coordinate
  embedding, Kalman filter tracking, Marzullo/geometric outlier rejection
- **layers/** — The 4-layer DHT data structures and valve transitions
  (L0→L1→L2→L3)
- **dht/** — Kademlia routing tables, per-layer key-value storage
- **epoch** — Three-phase epoch lifecycle (Measure → Solve → Publish)

### `rtfsn-net`
Networking abstraction over native UDP and WebRTC, protocol message types.

### `rtfsn-daemon`
Native daemon binary for running as a system service or cryptographic DMZ.

### `rtfsn-wasm`
WebAssembly bindings for browser-based participation via WebRTC data channels.

## Building

```bash
cargo build           # native
cargo test            # run all tests (43 tests)
```

## Key Concepts

**Holonomy defect detection**: A node with a bad clock produces inconsistent
twist entries detectable from any vantage point — no central authority needed.

**SAMR prime channels**: Independent channels at primes 2, 3, 5, 7 give
210 distinguishable states after 4 refinements. The channel with highest
defect is refined first (information-theoretically optimal).

**Temporal mirror**: The Layer 3 clock value is the fixed point where
predictive (forward extrapolation) and retrodictive (backward from boundary)
estimates agree — self-consistent regardless of temporal direction.

**Cryptographic DMZ**: The daemon can run locally as an isolated cryptographic
environment, bridging between the browser's internet-connected context and
local key material.
