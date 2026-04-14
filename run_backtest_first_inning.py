#!/usr/bin/env python3
"""
Run backtest for 1st inning models (moneyline + total over/under 0.5).
Collects historical data with 1st inning linescore, builds features, trains models.
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import requests
import time
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

from config.settings import (
    PROCESSED_DIR, OUTPUT_DIR, HISTORICAL_SEASONS,
    MLB_API_BASE, TEAM_ID_TO_ABBREV,
)
from src.features.first_inning import build_first_inning_features_historical
from src.model.first_inning import train_first_inning_ml, train_first_inning_total
from src.utils.logging import get_logger
from src.utils.dates import season_start, season_end

log = get_logger("backtest_first_inning")


def collect_season_with_1st_inning(season: int) -> pd.DataFrame:
    """Collect season data including 1st inning linescore."""
    start = season_start(season)
    end = season_end(season)

    log.info(f"Collecting {season} season with 1st inning data...")

    url = f"{MLB_API_BASE}/schedule"
    params = {
        "sportId": 1,
        "startDate": start.strftime("%Y-%m-%d"),
        "endDate": end.strftime("%Y-%m-%d"),
        "gameType": "R",
        "hydrate": "linescore",
    }

    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    games = []
    for date_entry in data.get("dates", []):
        for game in date_entry.get("games", []):
            status = game.get("status", {}).get("detailedState", "")
            if status != "Final":
                continue

            home = game["teams"]["home"]
            away = game["teams"]["away"]
            home_id = home["team"]["id"]
            away_id = away["team"]["id"]

            linescore = game.get("linescore", {})
            innings = linescore.get("innings", [])

            if not innings:
                continue

            # 1st inning scores
            i1 = innings[0]
            i1_home = i1.get("home", {}).get("runs", 0)
            i1_away = i1.get("away", {}).get("runs", 0)

            home_score = home.get("score", 0)
            away_score = away.get("score", 0)

            games.append({
                "game_id": game["gamePk"],
                "game_date": date_entry["date"],
                "season": season,
                "home_team_id": home_id,
                "away_team_id": away_id,
                "home_team": TEAM_ID_TO_ABBREV.get(home_id, "UNK"),
                "away_team": TEAM_ID_TO_ABBREV.get(away_id, "UNK"),
                "home_score": home_score,
                "away_score": away_score,
                "i1_home_runs": i1_home,
                "i1_away_runs": i1_away,
            })

    df = pd.DataFrame(games)
    if not df.empty:
        df["game_date"] = pd.to_datetime(df["game_date"])
        df = df.drop_duplicates(subset=["game_id"])
    log.info(f"  Collected {len(df)} games for {season}")
    time.sleep(1)
    return df


def main():
    log.info("=== MLB 1st Inning Backtest ===")

    # Step 1: Collect data
    cache_path = PROCESSED_DIR / "historical_games_first_inning.parquet"

    if cache_path.exists():
        games = pd.read_parquet(cache_path)
        log.info(f"Loaded {len(games)} games from cache")
    else:
        all_dfs = []
        for season in HISTORICAL_SEASONS:
            df = collect_season_with_1st_inning(season)
            if not df.empty:
                all_dfs.append(df)

        if not all_dfs:
            print("No data collected!")
            return

        games = pd.concat(all_dfs, ignore_index=True)
        games = games.sort_values("game_date").reset_index(drop=True)
        games.to_parquet(cache_path, index=False)
        log.info(f"Collected and cached {len(games)} games")

    # Stats overview
    total_1st_inn = games["i1_home_runs"] + games["i1_away_runs"]
    pct_scored = (total_1st_inn > 0).mean()
    avg_runs = total_1st_inn.mean()
    home_won_1st = ((games["i1_home_runs"] > games["i1_away_runs"]) |
                    ((games["i1_home_runs"] == 0) & (games["i1_away_runs"] == 0))).mean()
    log.info(f"1st inning stats: {pct_scored:.1%} games have scoring, avg {avg_runs:.2f} runs")

    # Step 2: Build features
    log.info("Building features...")
    features = build_first_inning_features_historical(games)

    # Add targets
    features["i1_home_won"] = (features["i1_home_runs"] > features["i1_away_runs"]).astype(int)
    features["i1_went_over"] = (features["i1_total"] > 0.5).astype(int)  # Over 0.5 = any run scored

    features.to_parquet(PROCESSED_DIR / "first_inning_training_dataset.parquet", index=False)
    log.info(f"Training dataset: {len(features)} games")

    # Step 3: Train models
    train_seasons = HISTORICAL_SEASONS

    # ML model: train only on decided games (ties push in market)
    decided = features[features["i1_home_runs"] != features["i1_away_runs"]].copy()
    log.info(f"\nDecided games (non-tie 1st innings): {len(decided)}")
    log.info(f"  Home win rate: {decided['i1_home_won'].mean():.1%}")

    log.info("\nTraining 1st Inning Moneyline model (decided games only)...")
    ml_model, ml_meta, ml_preds = train_first_inning_ml(decided, train_seasons)

    log.info("\nTraining 1st Inning Total model...")
    total_model, total_meta, total_preds = train_first_inning_total(features, train_seasons)

    # Step 4: Summary
    metrics = {}
    if ml_meta:
        metrics["first_inning_ml"] = {
            "accuracy": ml_meta.get("test_accuracy", 0),
            "test_games": ml_meta.get("test_games", 0),
        }
        log.info(f"\n1st Inning ML: {ml_meta.get('test_accuracy', 0):.1%} accuracy")
    if total_meta:
        metrics["first_inning_total"] = {
            "accuracy": total_meta.get("test_accuracy", 0),
            "test_games": total_meta.get("test_games", 0),
        }
        log.info(f"1st Inning Total: {total_meta.get('test_accuracy', 0):.1%} accuracy")

    with open(OUTPUT_DIR / "backtest_first_inning_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    log.info("\nBacktest complete! Models saved.")


if __name__ == "__main__":
    main()
