#!/usr/bin/env python3
"""Collect TCP connection-establishment latency to HTTPS hosts.

Reconstructed reference collector, not the original missing experiment script.
DNS is resolved before measurement. The measured duration is TCP connection
establishment time, not a pure network RTT; it can include local scheduling,
network-path, and server-side effects.
"""
from __future__ import annotations

import argparse
import csv
import os
import socket
import time
from datetime import datetime, timezone

SERVERS = [
    "cloudflare.com", "amazon.in", "google.com", "abc.net.au",
    "manipal.edu", "mit.edu", "bbc.co.uk", "github.com",
    "wikipedia.org", "yahoo.co.jp", "uct.ac.za", "gov.br",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_servers() -> dict[str, tuple[int, tuple]]:
    """Resolve each hostname once; retain the first TCP address returned."""
    resolved: dict[str, tuple[int, tuple]] = {}
    for host in SERVERS:
        try:
            answers = socket.getaddrinfo(
                host, 443, type=socket.SOCK_STREAM
            )
            if answers:
                family, socktype, proto, _, sockaddr = answers[0]
                resolved[host] = (family, sockaddr)
                print(f"Resolved {host} -> {sockaddr[0]}")
            else:
                print(f"DNS resolution returned no addresses for {host}")
        except OSError as exc:
            print(f"DNS resolution failed for {host}: {exc}")
    return resolved


def probe(address: tuple[int, tuple], timeout: float):
    family, sockaddr = address
    start = time.perf_counter()
    try:
        with socket.socket(family, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            sock.connect(sockaddr)
        return (time.perf_counter() - start) * 1000.0, ""
    except OSError as exc:
        return None, str(exc)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network-label", required=True,
                        help="e.g. wifi or hotspot")
    parser.add_argument("--duration-minutes", type=float, default=15)
    parser.add_argument("--interval-seconds", type=float, default=5)
    parser.add_argument("--timeout-seconds", type=float, default=3)
    parser.add_argument("--output", required=True)
    parser.add_argument("--load-flag", type=int, choices=(0, 1), default=0,
                        help="labels the observed load condition; does not generate load")
    args = parser.parse_args()

    if args.duration_minutes <= 0:
        parser.error("duration must be positive")
    if args.interval_seconds <= 0 or args.timeout_seconds <= 0:
        parser.error("interval and timeout must be positive")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    print("Resolving hosts once before measurement...")
    resolved = resolve_servers()
    if not resolved:
        parser.error("No hosts could be resolved; check the network and DNS")

    fields = [
        "timestamp", "probe_timestamp", "round_id", "server", "rtt_ms",
        "failed", "error", "network", "load_on",
    ]

    start = time.monotonic()
    end = start + args.duration_minutes * 60
    round_id = 0
    total_rows = 0
    overruns = 0

    with open(args.output, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        while True:
            scheduled_start = start + round_id * args.interval_seconds
            if scheduled_start >= end:
                break

            delay = scheduled_start - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            elif round_id > 0:
                overruns += 1

            round_id += 1
            round_timestamp = utc_now()

            for host in SERVERS:
                probe_timestamp = utc_now()
                if host in resolved:
                    latency, error = probe(resolved[host], args.timeout_seconds)
                else:
                    latency, error = None, "DNS resolution failed"

                writer.writerow({
                    "timestamp": round_timestamp,
                    "probe_timestamp": probe_timestamp,
                    "round_id": round_id,
                    "server": host,
                    "rtt_ms": "" if latency is None else round(latency, 3),
                    "failed": int(latency is None),
                    "error": error,
                    "network": args.network_label,
                    "load_on": args.load_flag,
                })
                total_rows += 1

            file.flush()
            if round_id % 12 == 0:
                print(f"Completed {round_id} rounds ({total_rows} probes)")

    print(f"Completed {round_id} rounds; wrote {total_rows} rows.")
    print(f"Rounds that started later than scheduled: {overruns}")
    print(f"Saved measurements to {args.output}")


if __name__ == "__main__":
    main()
