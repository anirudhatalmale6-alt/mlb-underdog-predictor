#!/usr/bin/env python3
"""
Backtest player props models (pitcher strikeouts + batter hits).
Walk-forward validation: train on prior seasons, test on current season.
"""

import sys
import os
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

from config.settings import MODELS_DIR, OUTPUT_DIR, ABBREV_TO_TEAM_ID, XGB_PARAMS
from src.ingest.props_collect import (
    get_season_starters, get_pitcher_game_log,
    get_batter_game_log, get_team_strikeout_rate,
)
from src.features.props import (
    PITCHER_K_FEATURES, BATTER_HITS_FEATURES, PITCHER_OUTS_FEATURES,
    build_pitcher_k_features, build_batter_hits_features, build_pitcher_outs_features,
)
from src.utils.logging import get_logger

log = get_logger("backtest_props")

MLB_API = "https://statsapi.mlb.com/api/v1"
SEASONS = [2021, 2022, 2023, 2024]
TEST_SEASONS = [2023, 2024]


def collect_pitcher_k_dataset(seasons: list[int]) -> pd.DataFrame:
    """Collect pitcher strikeout dataset across multiple seasons."""
    all_rows = []

    for season in seasons:
        log.info(f"Collecting pitcher K data for {season}...")
        games = get_season_starters(season)

        # Group games by starting pitcher
        pitcher_games = {}
        for g in games:
            for side in ["home", "away"]:
                sp_id = g.get(f"{side}_sp_id")
                if sp_id:
                    pitcher_games.setdefault(sp_id, []).append(g)

        # Get unique pitcher IDs with enough starts
        pitcher_ids = [pid for pid, gs in pitcher_games.items() if len(gs) >= 8]
        log.info(f"  {len(pitcher_ids)} qualified pitchers (8+ starts)")

        # Get game logs for each pitcher
        for i, pid in enumerate(pitcher_ids):
            if i % 50 == 0:
                log.info(f"  Processing pitcher {i+1}/{len(pitcher_ids)}...")

            game_log = get_pitcher_game_log(pid, season)
            if len(game_log) < 6:
                continue

            # Get opposing team K rates (cached per team per season)
            for idx in range(5, len(game_log)):
                game = game_log[idx]
                opp_id = game.get("opponent_id")
                opp_k_rate = _get_cached_team_k_rate(opp_id, season)

                features = build_pitcher_k_features(
                    game_log, idx,
                    opp_k_rate=opp_k_rate,
                    is_home=game.get("is_home", False),
                    min_starts=5,
                )
                if features is None:
                    continue

                actual_k = game["strikeouts"]
                # Proxy line: rolling average of K per start
                prior_ks = [g["strikeouts"] for g in game_log[:idx]]
                proxy_line = np.mean(prior_ks[-10:])

                features["actual_k"] = actual_k
                features["proxy_line"] = round(proxy_line, 1)
                features["went_over"] = 1 if actual_k > proxy_line else 0
                features["season"] = season
                features["pitcher_id"] = pid
                features["game_date"] = game["date"]

                all_rows.append(features)

    df = pd.DataFrame(all_rows)
    log.info(f"Total pitcher K samples: {len(df)}")
    return df


def collect_pitcher_outs_dataset(seasons: list[int]) -> pd.DataFrame:
    """Collect pitcher outs recorded dataset across multiple seasons."""
    all_rows = []

    def ip_to_outs(ip: float) -> int:
        full = int(ip)
        partial = ip - full
        return full * 3 + round(partial * 10)

    for season in seasons:
        log.info(f"Collecting pitcher outs data for {season}...")
        games = get_season_starters(season)

        pitcher_games = {}
        for g in games:
            for side in ["home", "away"]:
                sp_id = g.get(f"{side}_sp_id")
                if sp_id:
                    pitcher_games.setdefault(sp_id, []).append(g)

        pitcher_ids = [pid for pid, gs in pitcher_games.items() if len(gs) >= 8]
        log.info(f"  {len(pitcher_ids)} qualified pitchers (8+ starts)")

        for i, pid in enumerate(pitcher_ids):
            if i % 50 == 0:
                log.info(f"  Processing pitcher {i+1}/{len(pitcher_ids)}...")

            game_log = get_pitcher_game_log(pid, season)
            if len(game_log) < 6:
                continue

            for idx in range(5, len(game_log)):
                game = game_log[idx]

                features = build_pitcher_outs_features(
                    game_log, idx,
                    is_home=game.get("is_home", False),
                    min_starts=5,
                )
                if features is None:
                    continue

                actual_outs = ip_to_outs(game["ip"])
                prior_ips = [g["ip"] for g in game_log[:idx]]
                prior_outs = [ip_to_outs(ip) for ip in prior_ips]
                proxy_line = np.mean(prior_outs[-10:])

                features["actual_outs"] = actual_outs
                features["proxy_line"] = round(proxy_line, 1)
                features["went_over"] = 1 if actual_outs > proxy_line else 0
                features["season"] = season
                features["pitcher_id"] = pid
                features["game_date"] = game["date"]

                all_rows.append(features)

    df = pd.DataFrame(all_rows)
    log.info(f"Total pitcher outs samples: {len(df)}")
    return df


def collect_batter_hits_dataset(seasons: list[int]) -> pd.DataFrame:
    """Collect batter hits dataset across multiple seasons."""
    all_rows = []

    for season in seasons:
        log.info(f"Collecting batter hits data for {season}...")
        games = get_season_starters(season)

        # Get unique SP IDs and cache their stats
        sp_stats_cache = {}

        # Get all starting lineup batters from schedule
        # We'll use top batters by sampling from game starters
        # Simpler: get all team rosters and find batters with 300+ AB
        team_ids = set()
        for g in games:
            for t in [g.get("home_team", ""), g.get("away_team", "")]:
                tid = ABBREV_TO_TEAM_ID.get(t)
                if tid:
                    team_ids.add(tid)

        # Get top batters by AB from league leaders API
        all_batters = []
        try:
            url = f"{MLB_API}/stats/leaders?leaderCategories=atBats&season={season}&sportId=1&limit=150&statType=season"
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            leaders = r.json().get("leagueLeaders", [{}])[0].get("leaders", [])
            for l in leaders:
                ab = int(l.get("value", 0))
                if ab >= 300:
                    p = l.get("person", {})
                    all_batters.append({
                        "player_id": p.get("id"),
                        "name": p.get("fullName", ""),
                        "team_id": l.get("team", {}).get("id"),
                        "ab": ab,
                    })
            time.sleep(0.3)
        except Exception as e:
            log.warning(f"Error getting batting leaders for {season}: {e}")

        log.info(f"  {len(all_batters)} qualified batters (300+ AB)")

        for i, batter in enumerate(all_batters):
            if i % 30 == 0:
                log.info(f"  Processing batter {i+1}/{len(all_batters)}...")

            blog = get_batter_game_log(batter["player_id"], season)
            if len(blog) < 20:
                continue

            for idx in range(15, len(blog)):
                game = blog[idx]

                features = build_batter_hits_features(
                    blog, idx,
                    opp_sp_stats=None,  # No SP matchup in backtest
                    is_home=game.get("is_home", False),
                    min_games=15,
                )
                if features is None:
                    continue

                actual_hits = game["hits"]
                # Proxy line: rolling average hits per game
                prior_hits = [g["hits"] for g in blog[:idx]]
                proxy_line = np.mean(prior_hits[-15:])

                features["actual_hits"] = actual_hits
                features["proxy_line"] = round(proxy_line, 2)
                features["went_over"] = 1 if actual_hits > proxy_line else 0
                features["season"] = season
                features["batter_id"] = batter["player_id"]
                features["game_date"] = game["date"]

                all_rows.append(features)

    df = pd.DataFrame(all_rows)
    log.info(f"Total batter hits samples: {len(df)}")
    return df


# Cache for team K rates
_team_k_cache = {}

def _get_cached_team_k_rate(team_id: int, season: int) -> float:
    key = (team_id, season)
    if key not in _team_k_cache:
        _team_k_cache[key] = get_team_strikeout_rate(team_id, season)
    return _team_k_cache[key]


def walk_forward_train(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    model_name: str,
    test_seasons: list[int],
) -> dict:
    """Walk-forward validation: train on prior seasons, test on each test season."""
    all_preds = []
    all_actuals = []

    for test_season in test_seasons:
        train_df = df[df["season"] < test_season]
        test_df = df[df["season"] == test_season]

        if train_df.empty or test_df.empty:
            continue

        X_train = train_df[feature_cols].fillna(0)
        y_train = train_df[target_col]
        X_test = test_df[feature_cols].fillna(0)
        y_test = test_df[target_col]

        params = XGB_PARAMS.copy()
        model = XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, preds)
        log.info(f"  {model_name} {test_season}: {acc:.1%} ({len(test_df)} samples)")

        all_preds.extend(preds)
        all_actuals.extend(y_test)

    overall_acc = accuracy_score(all_actuals, all_preds)
    log.info(f"  {model_name} overall: {overall_acc:.1%} ({len(all_actuals)} samples)")

    # Train final model on all data
    X_all = df[feature_cols].fillna(0)
    y_all = df[target_col]
    final_model = XGBClassifier(**XGB_PARAMS)
    final_model.fit(X_all, y_all, verbose=False)

    # Save model
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = MODELS_DIR / f"{model_name}_{ts}.joblib"
    latest_path = MODELS_DIR / f"{model_name}_latest.joblib"
    meta_path = MODELS_DIR / f"{model_name}_latest_meta.json"

    joblib.dump(final_model, model_path)
    joblib.dump(final_model, latest_path)

    meta = {
        "model_name": model_name,
        "features": feature_cols,
        "trained_on": str(datetime.now()),
        "seasons": [int(s) for s in df["season"].unique()],
        "total_samples": int(len(df)),
        "accuracy": round(float(overall_acc), 4),
        "test_seasons": [int(s) for s in test_seasons],
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return {
        "model_name": model_name,
        "accuracy": overall_acc,
        "total_samples": len(all_actuals),
        "model_path": str(latest_path),
    }


def main():
    log.info("=" * 60)
    log.info("MLB Player Props Backtest")
    log.info("=" * 60)

    # ── Pitcher Strikeouts ──
    log.info("\n--- PITCHER STRIKEOUTS ---")
    k_df = collect_pitcher_k_dataset(SEASONS)

    if not k_df.empty:
        k_results = walk_forward_train(
            k_df, PITCHER_K_FEATURES, "went_over",
            "xgb_pitcher_k", TEST_SEASONS,
        )
        log.info(f"Pitcher K model: {k_results['accuracy']:.1%} accuracy")
    else:
        log.error("No pitcher K data collected")
        k_results = None

    # ── Pitcher Outs Recorded ──
    log.info("\n--- PITCHER OUTS RECORDED ---")
    o_df = collect_pitcher_outs_dataset(SEASONS)

    if not o_df.empty:
        o_results = walk_forward_train(
            o_df, PITCHER_OUTS_FEATURES, "went_over",
            "xgb_pitcher_outs", TEST_SEASONS,
        )
        log.info(f"Pitcher Outs model: {o_results['accuracy']:.1%} accuracy")
    else:
        log.error("No pitcher outs data collected")
        o_results = None

    # ── Batter Hits ──
    log.info("\n--- BATTER HITS ---")
    h_df = collect_batter_hits_dataset(SEASONS)

    if not h_df.empty:
        h_results = walk_forward_train(
            h_df, BATTER_HITS_FEATURES, "went_over",
            "xgb_batter_hits", TEST_SEASONS,
        )
        log.info(f"Batter Hits model: {h_results['accuracy']:.1%} accuracy")
    else:
        log.error("No batter hits data collected")
        h_results = None

    # Summary
    log.info("\n" + "=" * 60)
    log.info("BACKTEST RESULTS SUMMARY")
    log.info("=" * 60)
    if k_results:
        log.info(f"  Pitcher Strikeouts: {k_results['accuracy']:.1%} ({k_results['total_samples']} samples)")
    if o_results:
        log.info(f"  Pitcher Outs:       {o_results['accuracy']:.1%} ({o_results['total_samples']} samples)")
    if h_results:
        log.info(f"  Batter Hits:        {h_results['accuracy']:.1%} ({h_results['total_samples']} samples)")

    # Save metrics
    metrics = {
        "pitcher_k": k_results,
        "pitcher_outs": o_results,
        "batter_hits": h_results,
    }
    with open(OUTPUT_DIR / "backtest_props_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    log.info("Done!")


if __name__ == "__main__":
    main()
