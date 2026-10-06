//! NTP (RFC 5905) on the wire, sans-IO: a client request, a server's reply checked against it,
//! and NTP's 64-bit timestamps to and from nanoseconds since the Unix epoch.
//!
//! An NTP server is an absolute reference for the relational clock: one exchange gives the same
//! four timestamps as an exchange between two nodes (`t1`, `t4` on our clock, `t2`, `t3` on the
//! server's, which is UTC up to the server's own error), so a server is a peer whose timescale
//! is UTC. Its root distance (half its root delay plus its root dispersion) bounds how far that
//! timescale may sit from UTC; [`super::relational::RelationalClock::observe_reference`] turns it
//! into an [`super::relational::Anchor`].
//!
//! The request's transmit timestamp is an opaque cookie, not our clock (RFC 9109's data
//! minimisation): the server copies it into the reply's origin field, which is how the reply is
//! matched, and the caller keeps its own `t1`.

use serde::{Deserialize, Serialize};

/// The NTP port.
pub const PORT: u16 = 123;
/// An NTP packet without extensions.
pub const PACKET_LEN: usize = 48;
/// Seconds from 1900-01-01 (NTP era 0) to 1970-01-01.
pub const UNIX_TO_NTP_S: u64 = 2_208_988_800;
const ERA_S: u64 = 1 << 32;

/// Client mode.
pub const MODE_CLIENT: u8 = 3;
/// Server mode.
pub const MODE_SERVER: u8 = 4;
/// The leap indicator of an unsynchronised server.
pub const LEAP_UNSYNCHRONIZED: u8 = 3;

/// An NTP packet's header.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Packet {
    /// Leap indicator (0 none, 1 insert, 2 delete, 3 unsynchronised).
    pub leap: u8,
    /// Version (4).
    pub version: u8,
    /// Mode (3 client, 4 server).
    pub mode: u8,
    /// Stratum (0 kiss-o'-death, 1 primary, 2–15 secondary, 16 unsynchronised).
    pub stratum: u8,
    /// Poll exponent, log₂ s.
    pub poll: i8,
    /// Precision exponent, log₂ s.
    pub precision: i8,
    /// Round trip to the primary reference, NTP short format (16.16 s).
    pub root_delay: u32,
    /// Dispersion to the primary reference, NTP short format.
    pub root_dispersion: u32,
    /// Reference id (a refclock code at stratum 1, an address otherwise; the kiss code at 0).
    pub reference_id: [u8; 4],
    /// When the server's clock was last set.
    pub reference_ts: u64,
    /// The request's transmit timestamp, echoed.
    pub origin_ts: u64,
    /// When the request arrived (the server's `t2`).
    pub receive_ts: u64,
    /// When this packet left (the server's `t3`; a client's cookie).
    pub transmit_ts: u64,
}

/// Why bytes were not an acceptable reply.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NtpError {
    /// Shorter than a header.
    Short,
    /// Not a server-mode packet.
    NotServer(u8),
    /// The origin does not echo our cookie: late, duplicated or forged.
    WrongOrigin,
    /// Stratum 0: the server asks us to back off or go away (`RATE`, `DENY`, `RSTR`…).
    KissOfDeath([u8; 4]),
    /// The server says it is not synchronised (leap 3, or stratum 16 and above).
    Unsynchronized,
    /// A zero receive or transmit timestamp.
    ZeroTimestamp,
}

impl Packet {
    /// Serialise.
    pub fn encode(&self) -> [u8; PACKET_LEN] {
        let mut b = [0u8; PACKET_LEN];
        b[0] = (self.leap & 3) << 6 | (self.version & 7) << 3 | (self.mode & 7);
        b[1] = self.stratum;
        b[2] = self.poll as u8;
        b[3] = self.precision as u8;
        b[4..8].copy_from_slice(&self.root_delay.to_be_bytes());
        b[8..12].copy_from_slice(&self.root_dispersion.to_be_bytes());
        b[12..16].copy_from_slice(&self.reference_id);
        b[16..24].copy_from_slice(&self.reference_ts.to_be_bytes());
        b[24..32].copy_from_slice(&self.origin_ts.to_be_bytes());
        b[32..40].copy_from_slice(&self.receive_ts.to_be_bytes());
        b[40..48].copy_from_slice(&self.transmit_ts.to_be_bytes());
        b
    }

    /// Parse a header (extension fields and MACs after it are ignored).
    pub fn decode(b: &[u8]) -> Result<Packet, NtpError> {
        if b.len() < PACKET_LEN {
            return Err(NtpError::Short);
        }
        let u32_at = |i: usize| u32::from_be_bytes(b[i..i + 4].try_into().expect("four bytes"));
        let u64_at = |i: usize| u64::from_be_bytes(b[i..i + 8].try_into().expect("eight bytes"));
        Ok(Packet {
            leap: b[0] >> 6,
            version: (b[0] >> 3) & 7,
            mode: b[0] & 7,
            stratum: b[1],
            poll: b[2] as i8,
            precision: b[3] as i8,
            root_delay: u32_at(4),
            root_dispersion: u32_at(8),
            reference_id: b[12..16].try_into().expect("four bytes"),
            reference_ts: u64_at(16),
            origin_ts: u64_at(24),
            receive_ts: u64_at(32),
            transmit_ts: u64_at(40),
        })
    }
}

/// Nanoseconds since the Unix epoch as an NTP timestamp (the era is dropped, as on the wire).
pub fn to_ntp(unix_ns: u64) -> u64 {
    let secs = unix_ns / 1_000_000_000 + UNIX_TO_NTP_S;
    let frac = ((unix_ns % 1_000_000_000) << 32) / 1_000_000_000;
    (secs % ERA_S) << 32 | frac
}

/// An NTP timestamp as nanoseconds since the Unix epoch, in whichever era puts it nearest
/// `near_unix_ns` (the wire carries no era; any clock within 68 years resolves it).
pub fn from_ntp(ts: u64, near_unix_ns: u64) -> u64 {
    let secs = ts >> 32;
    let frac_ns = ((ts & 0xFFFF_FFFF) * 1_000_000_000 + (1 << 31)) >> 32;
    let near_ntp = near_unix_ns / 1_000_000_000 + UNIX_TO_NTP_S;
    let era = near_ntp / ERA_S;
    let best = [era.saturating_sub(1), era, era + 1]
        .into_iter()
        .map(|e| e * ERA_S + secs)
        .min_by_key(|s| s.abs_diff(near_ntp))
        .expect("three candidates");
    best.saturating_sub(UNIX_TO_NTP_S) * 1_000_000_000 + frac_ns
}

/// NTP short format (16.16 seconds) as nanoseconds.
pub fn short_to_ns(v: u32) -> u64 {
    ((v as u64) * 1_000_000_000) >> 16
}

/// A client request whose transmit timestamp is `cookie`.
pub fn request(cookie: u64) -> [u8; PACKET_LEN] {
    Packet {
        leap: 0,
        version: 4,
        mode: MODE_CLIENT,
        stratum: 0,
        poll: 4,
        precision: -20,
        root_delay: 0,
        root_dispersion: 0,
        reference_id: [0; 4],
        reference_ts: 0,
        origin_ts: 0,
        receive_ts: 0,
        transmit_ts: cookie,
    }
    .encode()
}

/// What a server's reply says, in nanoseconds since the Unix epoch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Reply {
    /// The server's clock when our request arrived.
    pub t2_ns: u64,
    /// The server's clock when the reply left.
    pub t3_ns: u64,
    /// Its stratum.
    pub stratum: u8,
    /// Its leap indicator.
    pub leap: u8,
    /// Its reference id.
    pub reference_id: [u8; 4],
    /// Its round trip to the primary reference, ns.
    pub root_delay_ns: u64,
    /// Its dispersion to the primary reference, ns.
    pub root_dispersion_ns: u64,
}

impl Reply {
    /// How far the server's clock may be from UTC: half its root delay plus its root dispersion.
    pub fn root_distance_ns(&self) -> f64 {
        self.root_delay_ns as f64 / 2.0 + self.root_dispersion_ns as f64
    }

    /// The reference id as text when it is one (a stratum-1 refclock such as `GPS` or `PPS`),
    /// else as a dotted address.
    pub fn reference(&self) -> String {
        let r = self.reference_id;
        if self.stratum <= 1 && r.iter().all(|c| c.is_ascii_graphic() || *c == 0) {
            r.iter()
                .take_while(|c| **c != 0)
                .map(|c| *c as char)
                .collect()
        } else {
            format!("{}.{}.{}.{}", r[0], r[1], r[2], r[3])
        }
    }
}

/// Check a reply against the cookie of the request it answers. `near_unix_ns` is any clock
/// within 68 years of now, to resolve NTP's era.
pub fn parse_reply(b: &[u8], cookie: u64, near_unix_ns: u64) -> Result<Reply, NtpError> {
    let p = Packet::decode(b)?;
    if p.mode != MODE_SERVER {
        return Err(NtpError::NotServer(p.mode));
    }
    if p.origin_ts != cookie {
        return Err(NtpError::WrongOrigin);
    }
    if p.stratum == 0 {
        return Err(NtpError::KissOfDeath(p.reference_id));
    }
    if p.leap == LEAP_UNSYNCHRONIZED || p.stratum >= 16 {
        return Err(NtpError::Unsynchronized);
    }
    if p.receive_ts == 0 || p.transmit_ts == 0 {
        return Err(NtpError::ZeroTimestamp);
    }
    Ok(Reply {
        t2_ns: from_ntp(p.receive_ts, near_unix_ns),
        t3_ns: from_ntp(p.transmit_ts, near_unix_ns),
        stratum: p.stratum,
        leap: p.leap,
        reference_id: p.reference_id,
        root_delay_ns: short_to_ns(p.root_delay),
        root_dispersion_ns: short_to_ns(p.root_dispersion),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2026-10-06T22:00:00.25Z.
    const NOW: u64 = 1_791_324_000_250_000_000;

    #[test]
    fn timestamps_round_trip_to_the_nanosecond_across_the_2036_era_boundary() {
        for ns in [
            NOW,
            NOW + 123_456_789,
            2_085_978_495_999_999_999,
            2_085_978_496_000_000_001,
            0,
        ] {
            let back = from_ntp(to_ntp(ns), ns.max(1));
            assert!(back.abs_diff(ns) <= 1, "{ns} → {back}");
        }
        // 2036-02-07T06:28:16Z is era 1, second 0: a clock just before it still resolves it.
        let era1 = 2_085_978_496_000_000_000u64;
        assert_eq!(to_ntp(era1) >> 32, 0);
        assert_eq!(from_ntp(to_ntp(era1), era1 - 60_000_000_000), era1);
    }

    #[test]
    fn a_request_is_mode_three_carrying_the_cookie() {
        let b = request(0xDEAD_BEEF_0123_4567);
        assert_eq!(b[0], 0x23); // LI 0, VN 4, mode 3
        let p = Packet::decode(&b).unwrap();
        assert_eq!(
            (p.mode, p.version, p.transmit_ts),
            (3, 4, 0xDEAD_BEEF_0123_4567)
        );
        assert_eq!(p.encode(), b);
    }

    fn server(cookie: u64, stratum: u8, leap: u8) -> [u8; PACKET_LEN] {
        Packet {
            leap,
            version: 4,
            mode: MODE_SERVER,
            stratum,
            poll: 6,
            precision: -23,
            root_delay: 0x0000_0800,      // 2048/65536 s = 31.25 ms
            root_dispersion: 0x0000_0040, // 64/65536 s ≈ 977 µs
            reference_id: *b"GPS\0",
            reference_ts: to_ntp(NOW - 16_000_000_000),
            origin_ts: cookie,
            receive_ts: to_ntp(NOW + 1_000_000),
            transmit_ts: to_ntp(NOW + 1_040_000),
        }
        .encode()
    }

    #[test]
    fn a_reply_is_checked_and_read() {
        let r = parse_reply(&server(77, 1, 0), 77, NOW).unwrap();
        assert!(r.t2_ns.abs_diff(NOW + 1_000_000) <= 1 && r.t3_ns.abs_diff(NOW + 1_040_000) <= 1);
        assert_eq!(r.reference(), "GPS");
        assert_eq!(r.root_delay_ns, 31_250_000);
        assert!((r.root_distance_ns() - (15_625_000.0 + 976_562.0)).abs() < 1.0);
        assert_eq!(
            parse_reply(&server(77, 1, 0), 78, NOW),
            Err(NtpError::WrongOrigin)
        );
        assert_eq!(
            parse_reply(&server(77, 0, 0), 77, NOW),
            Err(NtpError::KissOfDeath(*b"GPS\0"))
        );
        assert_eq!(
            parse_reply(&server(77, 2, 3), 77, NOW),
            Err(NtpError::Unsynchronized)
        );
        assert_eq!(
            parse_reply(&server(77, 16, 0), 77, NOW),
            Err(NtpError::Unsynchronized)
        );
        assert_eq!(
            parse_reply(&request(77), 77, NOW),
            Err(NtpError::NotServer(3))
        );
        assert_eq!(parse_reply(&[0u8; 20], 77, NOW), Err(NtpError::Short));
        let mut s2 = server(77, 2, 0);
        s2[12..16].copy_from_slice(&[10, 5, 0, 100]);
        assert_eq!(parse_reply(&s2, 77, NOW).unwrap().reference(), "10.5.0.100");
    }
}
