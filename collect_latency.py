#!/usr/bin/env python3
"""Collect TCP connection-latency measurements to a small set of HTTPS hosts.

This is a reconstructed reference collector based on the project report, not the
original missing collection script. socket.create_connection measures the time
for a TCP connection to complete; it is a practical approximation to handshake
RTT and may include local scheduling/server processing effects.
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


def probe(host: str, timeout: float) -> tuple[float | None, str | None]:
    start = time.perf_counter()
    try:
        with socket.create_connection((host, 443), timeout=timeout):
            elapsed_ms = (time.perf_counter() - start) * 1000.0
        return elapsed_ms, None
    except (OSError, TimeoutError) as exc:
        return None, str(exc)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--network-label", required=True, help="e.g. wifi or hotspot")
    p.add_argument("--duration-minutes", type=float, default=15)
    p.add_argument("--interval-seconds", type=float, default=5)
    p.add_argument("--timeout-seconds", type=float, default=3)
    p.add_argument("--output", required=True)
    p.add_argument("--load-flag", type=int, choices=(0, 1), default=0,
                   help="label all rows with this load state; does not generate load")
    args = p.parse_args()
    if args.duration_minutes <= 0 or args.interval_seconds <= 0:
        p.error("duration and interval must be positive")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    end_at = time.monotonic() + args.duration_minutes * 60
    fields = ["timestamp", "server", "rtt_ms", "failed", "error", "network", "load_on"]
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        while time.monotonic() < end_at:
            round_start = time.monotonic()
            stamp = datetime.now(timezone.utc).isoformat()
            for host in SERVERS:
                rtt, error = probe(host, args.timeout_seconds)
                writer.writerow({
                    "timestamp": stamp, "server": host,
                    "rtt_ms": "" if rtt is None else round(rtt, 3),
                    "failed": int(rtt is None), "error": error or "",
                    "network": args.network_label, "load_on": args.load_flag,
                })
                f.flush()
            remaining = args.interval_seconds - (time.monotonic() - round_start)
            if remaining > 0:
                time.sleep(min(remaining, max(0, end_at - time.monotonic())))
    print(f"Saved measurements to {args.output}")

if __name__ == "__main__":
    main()
