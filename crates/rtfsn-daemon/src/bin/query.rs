//! rtfsn-query: query clock statistics from a running rtfsn-daemon
//!
//! BUILD
//!   cargo build --release -p rtfsn-daemon
//!   sudo cp target/release/rtfsn-query /usr/local/bin/
//!
//! USAGE
//!   rtfsn-query                             # query localhost:4242
//!   rtfsn-query --target 192.168.1.100:4242
//!   rtfsn-query --watch 5                   # refresh every 5 s
//!   rtfsn-query --json                      # machine-readable JSON

use std::net::SocketAddr;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::Context;
use clap::Parser;
use tokio::net::UdpSocket;

use rtfsn_net::message::{PeerStats, ProtocolMessage};

#[derive(Parser, Debug)]
#[command(name = "rtfsn-query")]
#[command(about = "Query clock statistics from a running rtfsn-daemon")]
struct Args {
    /// Address of the daemon to query
    #[arg(short, long, default_value = "127.0.0.1:4242")]
    target: SocketAddr,

    /// Auto-refresh every N seconds (omit for one-shot)
    #[arg(short, long)]
    watch: Option<u64>,

    /// Emit raw JSON instead of human-readable output
    #[arg(long)]
    json: bool,

    /// Response wait timeout in milliseconds
    #[arg(long, default_value = "2000")]
    timeout_ms: u64,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let interval = args.watch.unwrap_or(0);

    loop {
        let query_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        match query_daemon(args.target, args.timeout_ms).await {
            Ok(resp) => {
                if args.json {
                    println!("{}", serde_json::to_string_pretty(&resp)?);
                } else {
                    print_stats(&resp, args.target, query_time);
                }
            }
            Err(e) => {
                eprintln!("error querying {}: {}", args.target, e);
                std::process::exit(1);
            }
        }

        if interval == 0 {
            break;
        }
        tokio::time::sleep(Duration::from_secs(interval)).await;
        println!();
    }

    Ok(())
}

async fn query_daemon(target: SocketAddr, timeout_ms: u64) -> anyhow::Result<ProtocolMessage> {
    let socket = UdpSocket::bind("0.0.0.0:0")
        .await
        .context("bind local UDP socket")?;

    let req = ProtocolMessage::StatsRequest {};
    let data = req.serialize().context("serialize StatsRequest")?;
    socket
        .send_to(&data, target)
        .await
        .context("send StatsRequest")?;

    let mut buf = vec![0u8; 65536];
    let (len, _) = tokio::time::timeout(
        Duration::from_millis(timeout_ms),
        socket.recv_from(&mut buf),
    )
    .await
    .map_err(|_| anyhow::anyhow!("timeout ({} ms) — is the daemon running?", timeout_ms))?
    .context("recv StatsResponse")?;

    ProtocolMessage::deserialize(&buf[..len]).context("deserialize StatsResponse")
}

// ─── formatting helpers ─────────────────────────────────────────────────────

fn fmt_ns(ns: u64) -> String {
    if ns == 0 {
        return "0 ns".to_string();
    }
    if ns < 1_000 {
        format!("{} ns", ns)
    } else if ns < 1_000_000 {
        format!("{:.2} us", ns as f64 / 1_000.0)
    } else if ns < 1_000_000_000 {
        format!("{:.3} ms", ns as f64 / 1_000_000.0)
    } else {
        format!("{:.3} s", ns as f64 / 1_000_000_000.0)
    }
}

fn fmt_ns_signed(ns: i64) -> String {
    let abs = ns.unsigned_abs();
    let sign = if ns < 0 { '-' } else { '+' };
    if abs < 1_000 {
        format!("{}{} ns", sign, abs)
    } else if abs < 1_000_000 {
        format!("{}{:.2} us", sign, abs as f64 / 1_000.0)
    } else if abs < 1_000_000_000 {
        format!("{}{:.3} ms", sign, abs as f64 / 1_000_000.0)
    } else {
        format!("{}{:.3} s", sign, abs as f64 / 1_000_000_000.0)
    }
}

fn fmt_rtt_col(ns: u64) -> String {
    // Fixed-width RTT suitable for table columns
    format!("{:>9}", fmt_ns(ns))
}

fn fmt_offset_col(ns: i64) -> String {
    format!("{:>11}", fmt_ns_signed(ns))
}

fn fmt_asym_col(ns: i64) -> String {
    format!("{:>9}", fmt_ns_signed(ns))
}

fn hms(unix_secs: u64) -> String {
    let h = (unix_secs % 86400) / 3600;
    let m = (unix_secs % 3600) / 60;
    let s = unix_secs % 60;
    format!("{:02}:{:02}:{:02} UTC", h, m, s)
}

fn peer_hex(id: &rtfsn_core::types::NodeId) -> String {
    id.0[..4].iter().map(|b| format!("{:02x}", b)).collect()
}

// ─── display ────────────────────────────────────────────────────────────────

fn print_stats(resp: &ProtocolMessage, target: SocketAddr, query_time: u64) {
    let ProtocolMessage::StatsResponse {
        node_id,
        epoch,
        offset_nanos,
        uncertainty_nanos,
        drift_ppb,
        solver_converged,
        solver_iters,
        max_defect_nanos,
        rtt_min_ns,
        rtt_mean_ns,
        rtt_max_ns,
        rtt_jitter_ns,
        geoid_depth,
        geoid_region_counts,
        chain_length,
        chain_valid,
        peers,
    } = resp
    else {
        eprintln!("unexpected response: {:?}", std::mem::discriminant(resp));
        return;
    };

    let sep = "═".repeat(64);
    let mid = "─".repeat(64);

    let node = peer_hex(node_id);

    println!("{}", sep);
    println!(
        " RTFSN node {}  epoch {}  @ {}  {}",
        node,
        epoch,
        target,
        hms(query_time)
    );
    println!("{}", mid);

    // Clock quality
    println!(
        "  offset     {:>13}  ±{}",
        fmt_ns_signed(*offset_nanos),
        fmt_ns(*uncertainty_nanos)
    );
    let drift_ns_per_s = *drift_ppb as f64 / 1_000.0;
    println!(
        "  drift      {:>+13.1} ppb  ({:+.2} ns/s)",
        drift_ppb, drift_ns_per_s
    );

    // RTT stats
    println!(
        "  RTT        min={}  mean={}  max={}",
        fmt_ns(*rtt_min_ns),
        fmt_ns(*rtt_mean_ns),
        fmt_ns(*rtt_max_ns)
    );
    println!("  jitter     {}  (RTT std dev across peers)", fmt_ns(*rtt_jitter_ns));

    // Solver
    let solver_status = if *solver_converged { "converged" } else { "timeout  " };
    println!(
        "  solver     {}  {} iters  defect={}",
        solver_status,
        solver_iters,
        fmt_ns(*max_defect_nanos)
    );

    // Geoid
    let region_str = if geoid_region_counts.is_empty() {
        "none".to_string()
    } else {
        geoid_region_counts
            .iter()
            .map(|c| c.to_string())
            .collect::<Vec<_>>()
            .join(" -> ")
    };
    println!("  geoid      {} levels  [{}]", geoid_depth, region_str);

    // Chain
    let chain_ok = if *chain_valid { "[OK]" } else { "[INVALID]" };
    println!("  chain      {} ticks  {}", chain_length, chain_ok);

    println!("{}", mid);

    // Peers table
    if peers.is_empty() {
        println!("  peers:  (none — waiting for first epoch exchange)");
    } else {
        println!("  peers ({}):", peers.len());
        println!(
            "  {:8}  {:24}  {:>9}  {:>11}  {:>9}  {:>4}  {:>4}  {:>9}  {}",
            "peer", "address", "RTT", "offset", "asym", "qual", "n", "jitter", "trend"
        );
        println!(
            "  {:8}  {:24}  {:9}  {:11}  {:9}  {:4}  {:4}  {:9}  {}",
            "--------",
            "------------------------",
            "---------",
            "-----------",
            "---------",
            "----",
            "----",
            "---------",
            "------------"
        );
        for p in peers {
            print_peer_row(p);
        }
    }

    println!("{}", sep);
}

fn print_peer_row(p: &PeerStats) {
    let hex = peer_hex(&p.peer_id);
    let addr = if p.peer_addr.len() > 24 {
        &p.peer_addr[..24]
    } else {
        &p.peer_addr
    };
    let trend = if p.trend_nanos_per_epoch == 0 {
        "stable".to_string()
    } else {
        format!("{:+}ns/ep", p.trend_nanos_per_epoch)
    };

    println!(
        "  {:8}  {:24}  {}  {}  {}  {:4}  {:4}  {:>9}  {}",
        hex,
        addr,
        fmt_rtt_col(p.rtt_nanos),
        fmt_offset_col(p.offset_nanos),
        fmt_asym_col(p.asymmetry_nanos),
        p.quality,
        p.sample_count,
        fmt_ns(p.rtt_jitter_ns),
        trend,
    );
}
