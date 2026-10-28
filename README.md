# Dataset

The original Wi-Fi and mobile-hotspot measurement CSVs were not present in the supplied project ZIP. Do not create or upload fabricated data as if they were the original measurements.

Place authentic measurement files here if recovered or recollected. Expected columns for the included analysis script:

- `timestamp`: ISO-8601 timestamp
- `server`: server/domain label
- `rtt_ms`: measured TCP connection latency in milliseconds; blank on failure
- `failed`: `1` for failed probe, otherwise `0`
- `load_on`: optional `1`/`0` indicator for background-load state

If measurements are newly collected, label them as a new dataset and update the paper to distinguish them from the original results.
