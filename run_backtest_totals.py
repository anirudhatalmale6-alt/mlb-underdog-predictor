#!/usr/bin/env python3
"""
Run backtest for MLB totals models (full game + F5).
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

from config.settings import PROCESSED_DIR, OUTPUT_DIR, HISTORICAL_SEASONS
from src.ingest.bulk_collect import collect_season_bulk
from src.features.totals import build_totals_features_historical
from src.model.totals import train_full_game_total, train_f5_total
from src.utils.logging import get_logger
from src.utils.dates import season_start, season_end

log = get_logger("backtest_totals")


def main():
    log.info("=== MLB Totals Backtest ===")

    # Step 1: Collect data with F5 scores
    log.info("Step 1: Collecting historical data with linescore...")
    cache_path = PROCESSED_DIR / "historical_games_with_totals.parquet"

    if cache_path.exists():
        games = pd.read_parquet(cache_path)
        log.info(f"  Loaded {len(games)} games from cache")
    else:
        all_dfs = []
        for season in HISTORICAL_SEASONS:
            df = collect_season_bulk(season)
            if not df.empty:
                df["season"] = season
                all_dfs.append(df)

        if not all_dfs:
            print("No data collected!")
            return

        games = pd.concat(all_dfs, ignore_index=True)
        games = games.sort_values("game_date").reset_index(drop=True)

        # Ensure we have total_runs and f5_total
        if "total_runs" not in games.columns:
            games["total_runs"] = games["home_score"] + games["away_score"]
        if "f5_total" not in games.columns:
            # Approximate F5 from full game if linescore wasn't available
            games["f5_total"] = (games["total_runs"] * 0.55).round().astype(int)

        games.to_parquet(cache_path, index=False)
        log.info(f"  Collected and cached {len(games)} games")

    # Step 2: Build features
    log.info("Step 2: Building totals features...")
    features = build_totals_features_historical(games)
    log.info(f"  Built {len(features)} feature vectors")

    # Step 3: Add targets
    # Use the per-game rolling combined scoring as the proxy line
    # This better simulates what the market line would be
    features["total_line_proxy"] = features["combined_rpg"].round(1)
    features["fg_went_over"] = (features["total_runs"] > features["total_line_proxy"]).astype(int)

    features["f5_line_proxy"] = features["combined_f5_rpg"].round(1)
    features["f5_went_over"] = (features["f5_total"] > features["f5_line_proxy"]).astype(int)

    features.to_parquet(PROCESSED_DIR / "totals_training_dataset.parquet", index=False)

    # Step 4: Train and evaluate
    train_seasons = HISTORICAL_SEASONS

    log.info("\nStep 3: Training Full Game Totals model...")
    fg_model, fg_meta, fg_preds = train_full_game_total(features, train_seasons)

    log.info("\nStep 4: Training F5 Innings Totals model...")
    f5_model, f5_meta, f5_preds = train_f5_total(features, train_seasons)

    # Step 5: Report
    print(f"\n{'='*60}")
    print(f"  MLB TOTALS BACKTEST RESULTS")
    print(f"{'='*60}")

    for label, preds, meta in [
        ("FULL GAME TOTALS", fg_preds, fg_meta),
        ("FIRST 5 INNINGS TOTALS", f5_preds, f5_meta),
    ]:
        if preds.empty:
            print(f"\n  {label}: No predictions generated")
            continue

        overall_acc = accuracy_score(preds["actual"], preds["model_pick"])
        print(f"\n  {label}")
        print(f"  Overall accuracy: {overall_acc:.1%}")
        print(f"  Games tested: {len(preds)}")

        for season in sorted(preds["season"].unique()):
            sp = preds[preds["season"] == season]
            sacc = accuracy_score(sp["actual"], sp["model_pick"])
            print(f"    {season}: {sacc:.1%} ({len(sp)} games)")

    print(f"\n{'='*60}")

    # Save report
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "full_game": {
            "accuracy": fg_meta.get("test_accuracy", 0),
            "n_games": fg_meta.get("test_games", 0),
        },
        "f5": {
            "accuracy": f5_meta.get("test_accuracy", 0),
            "n_games": f5_meta.get("test_games", 0),
        },
    }
    with open(OUTPUT_DIR / "backtest_totals_metrics.json", "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
