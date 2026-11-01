#!/usr/bin/env python3
"""Evaluate short-horizon TCP connection-latency predictors chronologically.

Reconstructed reference implementation. It is not the missing original
experiment code and cannot reproduce paper results without the original data.
The target is TCP connection-establishment time (ms), not pure network RTT.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

FEATURES = [
    "lag1", "lag2", "lag3",
    "roll_mean5", "roll_std5", "roll_min5", "roll_max5",
    "roll_mean12", "roll_std12", "ewma", "last_change",
    "load_on", "round_mean", "round_std", "fail_share",
    "server_median", "server_code",
]


def prepare_data(raw: pd.DataFrame) -> pd.DataFrame:
    required = {"timestamp", "server", "rtt_ms", "failed"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Input CSV missing required columns: {sorted(missing)}")

    d = raw.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce", utc=True)
    if "probe_timestamp" in d.columns:
        d["probe_timestamp"] = pd.to_datetime(
            d["probe_timestamp"], errors="coerce", utc=True
        )
    d["rtt_ms"] = pd.to_numeric(d["rtt_ms"], errors="coerce")
    d["failed"] = pd.to_numeric(d["failed"], errors="coerce").fillna(0).astype(int)
    d["failed"] = d["failed"].clip(0, 1)
    d["load_on"] = pd.to_numeric(
        d.get("load_on", pd.Series(0, index=d.index)), errors="coerce"
    ).fillna(0)

    d = d.dropna(subset=["timestamp", "server"]).copy()

    # New collector supplies round_id. For older CSVs, derive rounds from
    # the shared timestamp written for every server in a collection round.
    if "round_id" in d.columns and d["round_id"].notna().all():
        d["round_id"] = pd.to_numeric(d["round_id"], errors="coerce")
    else:
        round_keys = (
            d[["timestamp"]].drop_duplicates().sort_values("timestamp")
            .reset_index(drop=True)
        )
        round_keys["round_id"] = np.arange(1, len(round_keys) + 1)
        d = d.merge(round_keys, on="timestamp", how="left")

    d = d.dropna(subset=["round_id"])
    d["round_id"] = d["round_id"].astype(int)
    d = d.sort_values(["server", "round_id", "timestamp"]).drop_duplicates(
        ["server", "round_id"], keep="first"
    ).copy()

    # Historical features may carry the last successful RTT forward. Targets
    # always use the original RTT and are excluded when the future probe failed.
    by_server = d.groupby("server", sort=False)
    d["rtt_filled"] = by_server["rtt_ms"].ffill()

    for lag in (1, 2, 3):
        d[f"lag{lag}"] = by_server["rtt_filled"].shift(lag)

    for window in (5, 12):
        d[f"roll_mean{window}"] = by_server["rtt_filled"].transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).mean()
        )
        d[f"roll_std{window}"] = by_server["rtt_filled"].transform(
            lambda s: s.shift(1).rolling(window, min_periods=2).std()
        )

    d["roll_min5"] = by_server["rtt_filled"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).min()
    )
    d["roll_max5"] = by_server["rtt_filled"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).max()
    )
    d["ewma"] = by_server["rtt_filled"].transform(
        lambda s: s.shift(1).ewm(alpha=0.3, adjust=False).mean()
    )
    d["last_change"] = by_server["rtt_filled"].transform(
        lambda s: s.diff().shift(1)
    )
    d["server_median"] = by_server["rtt_filled"].transform(
        lambda s: s.shift(1).expanding().median()
    )
    d["fail_share"] = by_server["failed"].transform(
        lambda s: s.shift(1).rolling(12, min_periods=1).mean()
    )
    d["server_code"] = d["server"].astype("category").cat.codes

    # Cross-server context from the preceding round only.
    round_stats = d.groupby("round_id").agg(
        round_mean=("rtt_filled", "mean"),
        round_std=("rtt_filled", "std"),
    ).sort_index()
    round_stats[["round_mean", "round_std"]] = (
        round_stats[["round_mean", "round_std"]].shift(1)
    )
    d = d.join(round_stats, on="round_id")

    # Multi-step targets: h=1,3,6,12 correspond to 5,15,30,60 seconds
    # when the collection interval is five seconds.
    for horizon in (1, 3, 6, 12):
        d[f"target_h{horizon}"] = d.groupby("server")["rtt_ms"].shift(-horizon)
        d[f"target_failed_h{horizon}"] = (
            d.groupby("server")["failed"].shift(-horizon).fillna(1).astype(int)
        )
        d[f"target_round_h{horizon}"] = (
            d.groupby("server")["round_id"].shift(-horizon)
        )

    return d.sort_values(["round_id", "server"]).reset_index(drop=True)


def build_models():
    return {
        "Ridge": make_pipeline(
            SimpleImputer(), StandardScaler(), Ridge(alpha=1.0)
        ),
        "KNN": make_pipeline(
            SimpleImputer(), StandardScaler(), KNeighborsRegressor(n_neighbors=10)
        ),
        "DecisionTree": make_pipeline(
            SimpleImputer(),
            DecisionTreeRegressor(min_samples_leaf=20, random_state=42),
        ),
        "RandomForest": make_pipeline(
            SimpleImputer(),
            RandomForestRegressor(
                n_estimators=200, min_samples_leaf=5,
                random_state=42, n_jobs=-1,
            ),
        ),
        "GradientBoosting": make_pipeline(
            SimpleImputer(),
            HistGradientBoostingRegressor(
                max_iter=300, learning_rate=0.05, random_state=42
            ),
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Collected measurements CSV")
    parser.add_argument("--output", required=True, help="Output block-level metrics CSV")
    parser.add_argument("--gap-rounds", type=int, default=12)
    parser.add_argument("--blocks", type=int, default=5)
    parser.add_argument(
        "--horizons", default="1,3,6,12",
        help="Comma-separated forecast horizons in collection rounds",
    )
    args = parser.parse_args()

    if args.gap_rounds < 0 or args.blocks < 1:
        parser.error("gap-rounds must be non-negative and blocks must be positive")
    try:
        horizons = sorted({int(value.strip()) for value in args.horizons.split(",")})
    except ValueError:
        parser.error("horizons must be comma-separated positive integers")
    if not horizons or any(h < 1 for h in horizons):
        parser.error("all horizons must be positive integers")

    d = prepare_data(pd.read_csv(args.input))
    available_rounds = np.sort(d["round_id"].unique())
    if len(available_rounds) < args.blocks + args.gap_rounds + max(horizons) + 10:
        raise ValueError(
            "Not enough collection rounds for this evaluation. "
            "Collect more data or use fewer horizons/blocks."
        )

    initial_count = max(1, int(0.4 * len(available_rounds)))
    test_rounds = available_rounds[initial_count:]
    blocks = [b for b in np.array_split(test_rounds, args.blocks) if len(b)]
    models = build_models()
    records = []

    for horizon in horizons:
        target_col = f"target_h{horizon}"
        failed_col = f"target_failed_h{horizon}"
        target_round_col = f"target_round_h{horizon}"

        for block_no, block in enumerate(blocks, 1):
            if not len(block):
                continue
            block_start = int(block[0])
            # Keep a 12-round embargo before each test block and ensure the
            # training label itself is also strictly before that boundary.
            train_boundary = block_start - args.gap_rounds
            train = d[
                (d["round_id"] + horizon < train_boundary)
                & d[target_col].notna()
                & (d[failed_col] == 0)
            ]
            test = d[
                d["round_id"].isin(block)
                & d[target_col].notna()
                & (d[failed_col] == 0)
            ]

            if train.empty or test.empty:
                continue

            X_train, y_train = train[FEATURES], train[target_col]
            X_test, y_test = test[FEATURES], test[target_col]
            predictions = {
                "Persistence": test["lag1"].to_numpy(),
                "MovingAvg5": test["roll_mean5"].to_numpy(),
                "EWMA": test["ewma"].to_numpy(),
            }

            for name, model in models.items():
                model.fit(X_train, y_train)
                predictions[name] = model.predict(X_test)

            actual = y_test.to_numpy(dtype=float)
            for name, prediction in predictions.items():
                prediction = np.asarray(prediction, dtype=float)
                valid = np.isfinite(prediction) & np.isfinite(actual)
                if not valid.any():
                    continue
                y = actual[valid]
                y_hat = prediction[valid]
                abs_error = np.abs(y - y_hat)
                records.append({
                    "horizon_rounds": horizon,
                    "horizon_seconds_nominal": horizon * 5,
                    "block": block_no,
                    "model": name,
                    "n_test": int(valid.sum()),
                    "MAE_ms": mean_absolute_error(y, y_hat),
                    "RMSE_ms": mean_squared_error(y, y_hat) ** 0.5,
                    "MAPE_pct": float(
                        np.mean(abs_error / np.maximum(np.abs(y), 1e-9)) * 100
                    ),
                    "R2": r2_score(y, y_hat) if len(y) > 1 else np.nan,
                    "P95_abs_error_ms": float(np.percentile(abs_error, 95)),
                })

    if not records:
        raise ValueError(
            "No metrics generated. Check data quality and collect more rounds."
        )

    output = pd.DataFrame(records)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)

    summary = output.groupby(
        ["horizon_rounds", "model"]
    )[["MAE_ms", "RMSE_ms", "MAPE_pct", "R2", "P95_abs_error_ms"]].mean()
    print(summary.round(3).to_string())
    print(f"\nSaved block-level metrics to {output_path}")


if __name__ == "__main__":
    main()
