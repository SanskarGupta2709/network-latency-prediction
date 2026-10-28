#!/usr/bin/env python3
"""Compare simple and learned RTT predictors using chronological test blocks.

Reconstructed reference implementation based on the project report. It does not
claim exact reproduction of the paper's reported numbers without the original data.
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


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    required = {"timestamp", "server", "rtt_ms", "failed"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV missing required columns: {sorted(missing)}")
    d = df.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce", utc=True)
    d["rtt_ms"] = pd.to_numeric(d["rtt_ms"], errors="coerce")
    d["failed"] = pd.to_numeric(d["failed"], errors="coerce").fillna(0).astype(int)
    if "load_on" not in d:
        d["load_on"] = 0
    d["load_on"] = pd.to_numeric(d["load_on"], errors="coerce").fillna(0)
    d = d.dropna(subset=["timestamp", "server"]).sort_values(["server", "timestamp"])
    # Carry last valid RTT forward within each server for historical features only.
    d["rtt_filled"] = d.groupby("server")["rtt_ms"].ffill()
    grouped = d.groupby("server", group_keys=False)
    for lag in (1, 2, 3):
        d[f"lag{lag}"] = grouped["rtt_filled"].shift(lag)
    for window in (5, 12):
        d[f"roll_mean{window}"] = grouped["rtt_filled"].transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).mean())
        d[f"roll_std{window}"] = grouped["rtt_filled"].transform(
            lambda s: s.shift(1).rolling(window, min_periods=2).std())
    d["roll_min5"] = grouped["rtt_filled"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).min())
    d["roll_max5"] = grouped["rtt_filled"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).max())
    d["last_change"] = grouped["rtt_filled"].diff().shift(1)
    d["ewma"] = grouped["rtt_filled"].transform(lambda s: s.shift(1).ewm(alpha=0.3, adjust=False).mean())
    d["server_median"] = grouped["rtt_filled"].transform(lambda s: s.shift(1).expanding().median())
    d["server_code"] = d["server"].astype("category").cat.codes
    d["round_mean"] = d.groupby("timestamp")["rtt_filled"].transform("mean")
    d["round_std"] = d.groupby("timestamp")["rtt_filled"].transform("std")
    d["target"] = grouped["rtt_ms"].shift(-1)
    d["target_failed"] = grouped["failed"].shift(-1).fillna(1).astype(int)
    # Ensure contextual values are from the prior round, not the current target round.
    d["round_mean"] = d.groupby("server")["round_mean"].shift(1)
    d["round_std"] = d.groupby("server")["round_std"].shift(1)
    return d


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="CSV of collected measurements")
    p.add_argument("--output", required=True, help="Path for model metrics CSV")
    p.add_argument("--gap-rounds", type=int, default=12)
    p.add_argument("--blocks", type=int, default=5)
    args = p.parse_args()
    raw = pd.read_csv(args.input)
    d = make_features(raw)
    feature_cols = [c for c in d.columns if c.startswith(("lag", "roll_", "ewma", "server_", "round_", "last_change"))] + ["load_on"]
    feature_cols = list(dict.fromkeys(feature_cols))
    d = d.sort_values("timestamp").reset_index(drop=True)
    times = np.array(sorted(d["timestamp"].dropna().unique()))
    if len(times) < args.blocks + args.gap_rounds + 10:
        raise ValueError("Not enough distinct timestamps for the requested chronological evaluation.")
    initial = max(1, int(0.4 * len(times)))
    test_times = times[initial:]
    blocks = [b for b in np.array_split(test_times, args.blocks) if len(b)]
    models = {
        "Ridge": make_pipeline(SimpleImputer(), StandardScaler(), Ridge(alpha=1.0)),
        "KNN": make_pipeline(SimpleImputer(), StandardScaler(), KNeighborsRegressor(n_neighbors=10)),
        "DecisionTree": make_pipeline(SimpleImputer(), DecisionTreeRegressor(min_samples_leaf=20, random_state=42)),
        "RandomForest": make_pipeline(SimpleImputer(), RandomForestRegressor(n_estimators=200, min_samples_leaf=5, random_state=42, n_jobs=-1)),
        "GradientBoosting": make_pipeline(SimpleImputer(), HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=42)),
    }
    records = []
    for block_no, block in enumerate(blocks, 1):
        if not len(block):
            continue
        block_start = pd.Timestamp(block[0])
        train_cutoff = block_start - pd.Timedelta(seconds=args.gap_rounds * 5)
        train = d[(d.timestamp < train_cutoff) & d.target.notna() & (d.target_failed == 0)]
        test = d[d.timestamp.isin(block) & d.target.notna() & (d.target_failed == 0)]
        if train.empty or test.empty:
            continue
        X_train, y_train = train[feature_cols], train["target"]
        X_test, y_test = test[feature_cols], test["target"]
        preds = {
            "Persistence": test["lag1"].to_numpy(),
            "MovingAvg5": test["roll_mean5"].to_numpy(),
            "EWMA": test["ewma"].to_numpy(),
        }
        for name, model in models.items():
            model.fit(X_train, y_train)
            preds[name] = model.predict(X_test)
        for name, pred in preds.items():
            valid = np.isfinite(pred) & np.isfinite(y_test.to_numpy())
            if not valid.any():
                continue
            y, yh = y_test.to_numpy()[valid], pred[valid]
            records.append({"block": block_no, "model": name, "n_test": int(valid.sum()),
                            "MAE_ms": mean_absolute_error(y, yh),
                            "RMSE_ms": mean_squared_error(y, yh) ** 0.5,
                            "MAPE_pct": float(np.mean(np.abs((y-yh) / np.maximum(np.abs(y), 1e-9))) * 100),
                            "R2": r2_score(y, yh) if len(y) > 1 else np.nan,
                            "P95_abs_error_ms": float(np.percentile(np.abs(y-yh), 95))})
    if not records:
        raise ValueError("No metrics generated. Check CSV values and whether there are enough valid samples.")
    out = pd.DataFrame(records)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(out.groupby("model")[["MAE_ms", "RMSE_ms", "MAPE_pct", "R2", "P95_abs_error_ms"]].mean().round(3).to_string())
    print(f"\nSaved block-level metrics to {args.output}")

if __name__ == "__main__":
    main()
