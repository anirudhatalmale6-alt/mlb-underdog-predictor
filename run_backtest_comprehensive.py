#!/usr/bin/env python3
"""
Comprehensive backtest across all markets:
1. ML underdogs (with handicapping filters)
2. Full game totals (over/under)
3. Run line (+1.5 / -1.5)
4. Team totals (individual team over/under)
5. F5 innings: ML, totals, run line
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import requests
import time
from datetime import date
from collections import defaultdict
from sklearn.metrics import accuracy_score

from config.settings import (
    PROCESSED_DIR, OUTPUT_DIR, MLB_API_BASE, TEAM_ID_TO_ABBREV,
    HISTORICAL_SEASONS, PARK_FACTORS,
)
from src.utils.logging import get_logger

log = get_logger("backtest_comprehensive")

ROLLING_WINDOW = 30  # games for rolling stats


def collect_games_with_linescore(seasons: list[int]) -> pd.DataFrame:
    """Collect all games with full inning-by-inning linescore."""
    cache_path = PROCESSED_DIR / "historical_games_comprehensive.parquet"
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        log.info(f"Loaded {len(df)} games from cache")
        return df

    from src.utils.dates import season_start, season_end

    all_games = []
    for season in seasons:
        start = season_start(season)
        end = season_end(season)
        log.info(f"Collecting {season}...")

        resp = requests.get(f"{MLB_API_BASE}/schedule", params={
            "sportId": 1,
            "startDate": start.strftime("%Y-%m-%d"),
            "endDate": end.strftime("%Y-%m-%d"),
            "gameType": "R",
            "hydrate": "linescore",
        }, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        for date_entry in data.get("dates", []):
            for game in date_entry.get("games", []):
                if game.get("status", {}).get("detailedState") != "Final":
                    continue

                home = game["teams"]["home"]
                away = game["teams"]["away"]
                home_id = home["team"]["id"]
                away_id = away["team"]["id"]

                linescore = game.get("linescore", {})
                innings = linescore.get("innings", [])
                if len(innings) < 5:
                    continue

                home_score = home.get("score", 0)
                away_score = away.get("score", 0)

                # Inning-by-inning
                i1_home = innings[0].get("home", {}).get("runs", 0) if len(innings) > 0 else 0
                i1_away = innings[0].get("away", {}).get("runs", 0) if len(innings) > 0 else 0

                f5_home = sum(inn.get("home", {}).get("runs", 0) for inn in innings[:5])
                f5_away = sum(inn.get("away", {}).get("runs", 0) for inn in innings[:5])

                all_games.append({
                    "game_id": game["gamePk"],
                    "game_date": date_entry["date"],
                    "season": season,
                    "home_team_id": home_id,
                    "away_team_id": away_id,
                    "home_team": TEAM_ID_TO_ABBREV.get(home_id, "UNK"),
                    "away_team": TEAM_ID_TO_ABBREV.get(away_id, "UNK"),
                    "home_score": home_score,
                    "away_score": away_score,
                    "total_runs": home_score + away_score,
                    "home_win": 1 if home_score > away_score else 0,
                    "margin": home_score - away_score,
                    # 1st inning
                    "i1_home": i1_home,
                    "i1_away": i1_away,
                    "i1_total": i1_home + i1_away,
                    # First 5 innings
                    "f5_home": f5_home,
                    "f5_away": f5_away,
                    "f5_total": f5_home + f5_away,
                    "f5_home_win": 1 if f5_home > f5_away else (0 if f5_home < f5_away else -1),
                    "f5_margin": f5_home - f5_away,
                    "num_innings": len(innings),
                })

        time.sleep(1)

    df = pd.DataFrame(all_games)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values("game_date").reset_index(drop=True)
    df = df.drop_duplicates(subset=["game_id"])

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    log.info(f"Collected {len(df)} games total")
    return df


def build_rolling_stats(games: pd.DataFrame) -> pd.DataFrame:
    """Build rolling team statistics for each game (strictly before)."""
    # Sort chronologically
    games = games.sort_values("game_date").reset_index(drop=True)

    # Track per-team stats
    team_games = defaultdict(list)

    features = []
    for idx, row in games.iterrows():
        home = row["home_team"]
        away = row["away_team"]

        # Get rolling stats for both teams (from PRIOR games only)
        home_stats = _compute_rolling(team_games[home], home, row)
        away_stats = _compute_rolling(team_games[away], away, row)

        feat = {**row.to_dict()}
        # Home team rolling
        feat["home_rpg"] = home_stats["rpg"]
        feat["home_rapg"] = home_stats["rapg"]
        feat["home_win_pct"] = home_stats["win_pct"]
        feat["home_last10_pct"] = home_stats["last10_pct"]
        feat["home_streak"] = home_stats["streak"]
        feat["home_total_avg"] = home_stats["total_avg"]
        feat["home_team_total_avg"] = home_stats["team_total_avg"]
        feat["home_f5_avg"] = home_stats["f5_avg"]
        feat["home_f5_team_avg"] = home_stats["f5_team_avg"]
        feat["home_i1_scored_pct"] = home_stats["i1_scored_pct"]
        feat["home_i1_allowed_pct"] = home_stats["i1_allowed_pct"]

        # Away team rolling
        feat["away_rpg"] = away_stats["rpg"]
        feat["away_rapg"] = away_stats["rapg"]
        feat["away_win_pct"] = away_stats["win_pct"]
        feat["away_last10_pct"] = away_stats["last10_pct"]
        feat["away_streak"] = away_stats["streak"]
        feat["away_total_avg"] = away_stats["total_avg"]
        feat["away_team_total_avg"] = away_stats["team_total_avg"]
        feat["away_f5_avg"] = away_stats["f5_avg"]
        feat["away_f5_team_avg"] = away_stats["f5_team_avg"]
        feat["away_i1_scored_pct"] = away_stats["i1_scored_pct"]
        feat["away_i1_allowed_pct"] = away_stats["i1_allowed_pct"]

        # Combined / derived
        feat["combined_rpg"] = feat["home_rpg"] + feat["away_rpg"]
        feat["combined_rapg"] = feat["home_rapg"] + feat["away_rapg"]
        feat["park_factor"] = PARK_FACTORS.get(home, 100) / 100.0

        # Proxy lines (what a sportsbook would roughly set)
        feat["total_line_proxy"] = round((feat["home_rpg"] + feat["away_rapg"] +
                                          feat["away_rpg"] + feat["home_rapg"]) / 2 *
                                         feat["park_factor"], 1)
        feat["home_team_total_proxy"] = round((feat["home_rpg"] + feat["away_rapg"]) / 2 *
                                               feat["park_factor"], 1)
        feat["away_team_total_proxy"] = round((feat["away_rpg"] + feat["home_rapg"]) / 2, 1)
        feat["f5_total_proxy"] = round(feat["total_line_proxy"] * 0.55, 1)
        feat["f5_home_total_proxy"] = round(feat["home_team_total_proxy"] * 0.55, 1)
        feat["f5_away_total_proxy"] = round(feat["away_team_total_proxy"] * 0.55, 1)

        features.append(feat)

        # NOW update rolling data (after using it, to prevent leakage)
        is_home_game = True
        team_games[home].append({
            "runs_scored": row["home_score"], "runs_allowed": row["away_score"],
            "won": row["home_win"], "total": row["total_runs"],
            "team_score": row["home_score"],
            "f5_total": row["f5_total"], "f5_team": row["f5_home"],
            "i1_scored": 1 if row["i1_home"] > 0 else 0,
            "i1_allowed": 1 if row["i1_away"] > 0 else 0,
        })
        team_games[away].append({
            "runs_scored": row["away_score"], "runs_allowed": row["home_score"],
            "won": 1 - row["home_win"], "total": row["total_runs"],
            "team_score": row["away_score"],
            "f5_total": row["f5_total"], "f5_team": row["f5_away"],
            "i1_scored": 1 if row["i1_away"] > 0 else 0,
            "i1_allowed": 1 if row["i1_home"] > 0 else 0,
        })

    return pd.DataFrame(features)


def _compute_rolling(history: list, team: str, current_game) -> dict:
    """Compute rolling stats from team history."""
    defaults = {
        "rpg": 4.5, "rapg": 4.5, "win_pct": 0.5, "last10_pct": 0.5,
        "streak": 0, "total_avg": 9.0, "team_total_avg": 4.5,
        "f5_avg": 5.0, "f5_team_avg": 2.5,
        "i1_scored_pct": 0.3, "i1_allowed_pct": 0.3,
    }
    if len(history) < 10:
        return defaults

    recent = history[-ROLLING_WINDOW:]
    n = len(recent)

    rpg = sum(g["runs_scored"] for g in recent) / n
    rapg = sum(g["runs_allowed"] for g in recent) / n
    win_pct = sum(g["won"] for g in recent) / n

    last10 = history[-10:]
    last10_pct = sum(g["won"] for g in last10) / 10

    # Streak
    streak = 0
    for g in reversed(history):
        if g["won"] == 1:
            if streak >= 0:
                streak += 1
            else:
                break
        else:
            if streak <= 0:
                streak -= 1
            else:
                break

    total_avg = sum(g["total"] for g in recent) / n
    team_total_avg = sum(g["team_score"] for g in recent) / n
    f5_avg = sum(g["f5_total"] for g in recent) / n
    f5_team_avg = sum(g["f5_team"] for g in recent) / n
    i1_scored_pct = sum(g["i1_scored"] for g in recent) / n
    i1_allowed_pct = sum(g["i1_allowed"] for g in recent) / n

    return {
        "rpg": rpg, "rapg": rapg, "win_pct": win_pct, "last10_pct": last10_pct,
        "streak": streak, "total_avg": total_avg, "team_total_avg": team_total_avg,
        "f5_avg": f5_avg, "f5_team_avg": f5_team_avg,
        "i1_scored_pct": i1_scored_pct, "i1_allowed_pct": i1_allowed_pct,
    }


def apply_handicapping_filters(row) -> bool:
    """Apply handicapping filters for ML underdog picks."""
    # Determine underdog (team with lower win_pct is underdog)
    if row["home_win_pct"] < row["away_win_pct"]:
        ud_win_pct = row["home_win_pct"]
        fav_win_pct = row["away_win_pct"]
        ud_last10 = row["home_last10_pct"]
        ud_streak = row["home_streak"]
    else:
        ud_win_pct = row["away_win_pct"]
        fav_win_pct = row["home_win_pct"]
        ud_last10 = row["away_last10_pct"]
        ud_streak = row["away_streak"]

    if ud_win_pct < 0.450:
        return False
    if fav_win_pct > 0.550:
        return False
    if ud_last10 < 0.40:
        return False
    if ud_streak <= -4:
        return False
    return True


def backtest_ml_underdogs(df: pd.DataFrame) -> dict:
    """Backtest ML underdog picks with handicapping filters."""
    # Underdog = team with lower win_pct
    df = df.copy()
    df["underdog_is_home"] = df["home_win_pct"] < df["away_win_pct"]
    df["underdog_won"] = df.apply(
        lambda r: r["home_win"] if r["underdog_is_home"] else 1 - r["home_win"], axis=1
    )

    # Apply filters
    mask = df.apply(apply_handicapping_filters, axis=1)
    filtered = df[mask]

    if len(filtered) == 0:
        return {"market": "ML Underdogs (filtered)", "games": 0}

    wins = filtered["underdog_won"].sum()
    total = len(filtered)
    pct = wins / total

    # Without filters for comparison
    all_wins = df["underdog_won"].sum()
    all_total = len(df)
    all_pct = all_wins / all_total

    return {
        "market": "ML Underdogs",
        "with_filters": {"wins": int(wins), "losses": int(total - wins), "total": int(total),
                          "win_pct": round(pct * 100, 1)},
        "without_filters": {"wins": int(all_wins), "losses": int(all_total - all_wins),
                            "total": int(all_total), "win_pct": round(all_pct * 100, 1)},
    }


def backtest_full_game_totals(df: pd.DataFrame) -> dict:
    """Backtest full game over/under."""
    df = df.copy()
    proxy = df["total_line_proxy"]
    actual = df["total_runs"]

    # Exclude pushes
    non_push = actual != proxy
    went_over = actual[non_push] > proxy[non_push]

    over_pct = went_over.mean()
    # Simple model: predict over when combined_rpg > proxy line, under otherwise
    predicted_over = df.loc[non_push, "combined_rpg"] > proxy[non_push]
    accuracy = (predicted_over == went_over).mean()

    return {
        "market": "Full Game Totals (O/U)",
        "games": int(non_push.sum()),
        "pushes": int((~non_push).sum()),
        "over_pct": round(over_pct * 100, 1),
        "model_accuracy": round(accuracy * 100, 1),
    }


def backtest_run_line(df: pd.DataFrame) -> dict:
    """Backtest run line (-1.5 / +1.5)."""
    df = df.copy()
    # Favorite = team with higher win_pct
    # Standard run line: favorite -1.5, underdog +1.5
    fav_is_home = df["home_win_pct"] >= df["away_win_pct"]

    fav_margin = np.where(fav_is_home, df["margin"], -df["margin"])
    fav_covers = fav_margin > 1.5  # Favorite wins by 2+
    ud_covers = fav_margin < -1.5  # Underdog wins by 2+ (or just wins)

    # Actually for run line: fav needs to win by 2+, underdog covers if they lose by 1 or win
    ud_covers_rl = fav_margin < 1.5  # Underdog +1.5 covers

    fav_cover_pct = fav_covers.mean()
    ud_cover_pct = ud_covers_rl.mean()

    # Model: predict favorite -1.5 when their win_pct is much higher
    fav_wp = np.where(fav_is_home, df["home_win_pct"], df["away_win_pct"])
    ud_wp = np.where(fav_is_home, df["away_win_pct"], df["home_win_pct"])
    wp_diff = fav_wp - ud_wp

    # Predict fav covers when wp_diff > 0.05 (sizable edge)
    pred_fav_covers = wp_diff > 0.05
    accuracy = (pred_fav_covers == fav_covers).mean()

    return {
        "market": "Run Line (-1.5/+1.5)",
        "games": int(len(df)),
        "fav_cover_pct": round(fav_cover_pct * 100, 1),
        "ud_cover_pct": round(ud_cover_pct * 100, 1),
        "model_accuracy": round(accuracy * 100, 1),
    }


def backtest_team_totals(df: pd.DataFrame) -> dict:
    """Backtest team totals (individual team over/under)."""
    df = df.copy()

    # Home team totals
    home_proxy = df["home_team_total_proxy"]
    home_non_push = df["home_score"] != home_proxy
    home_over = df.loc[home_non_push, "home_score"] > home_proxy[home_non_push]
    home_pred = df.loc[home_non_push, "home_rpg"] > home_proxy[home_non_push]
    home_acc = (home_pred == home_over).mean()

    # Away team totals
    away_proxy = df["away_team_total_proxy"]
    away_non_push = df["away_score"] != away_proxy
    away_over = df.loc[away_non_push, "away_score"] > away_proxy[away_non_push]
    away_pred = df.loc[away_non_push, "away_rpg"] > away_proxy[away_non_push]
    away_acc = (away_pred == away_over).mean()

    return {
        "market": "Team Totals (O/U)",
        "home_games": int(home_non_push.sum()),
        "home_accuracy": round(home_acc * 100, 1),
        "away_games": int(away_non_push.sum()),
        "away_accuracy": round(away_acc * 100, 1),
        "combined_accuracy": round((home_acc + away_acc) / 2 * 100, 1),
    }


def backtest_f5_ml(df: pd.DataFrame) -> dict:
    """Backtest First 5 innings moneyline."""
    df = df.copy()
    # Exclude ties (pushes in F5 ML)
    decided = df[df["f5_home_win"] != -1]

    home_win_pct = (decided["f5_home_win"] == 1).mean()

    # Model: predict home wins F5 when home_win_pct higher
    pred_home = decided["home_win_pct"] > decided["away_win_pct"]
    actual_home = decided["f5_home_win"] == 1
    accuracy = (pred_home == actual_home).mean()

    return {
        "market": "F5 Innings ML",
        "decided_games": int(len(decided)),
        "ties": int(len(df) - len(decided)),
        "tie_pct": round((len(df) - len(decided)) / len(df) * 100, 1),
        "home_win_pct": round(home_win_pct * 100, 1),
        "model_accuracy": round(accuracy * 100, 1),
    }


def backtest_f5_totals(df: pd.DataFrame) -> dict:
    """Backtest First 5 innings totals."""
    df = df.copy()
    proxy = df["f5_total_proxy"]
    non_push = df["f5_total"] != proxy
    went_over = df.loc[non_push, "f5_total"] > proxy[non_push]

    over_pct = went_over.mean()
    pred_over = df.loc[non_push, "combined_rpg"] * 0.55 > proxy[non_push]
    accuracy = (pred_over == went_over).mean()

    return {
        "market": "F5 Innings Totals (O/U)",
        "games": int(non_push.sum()),
        "pushes": int((~non_push).sum()),
        "over_pct": round(over_pct * 100, 1),
        "model_accuracy": round(accuracy * 100, 1),
    }


def backtest_f5_run_line(df: pd.DataFrame) -> dict:
    """Backtest First 5 innings run line (+0.5 / -0.5)."""
    df = df.copy()
    # F5 run line is typically -0.5 / +0.5
    fav_is_home = df["home_win_pct"] >= df["away_win_pct"]
    fav_f5_margin = np.where(fav_is_home, df["f5_margin"], -df["f5_margin"])

    fav_covers = fav_f5_margin > 0.5  # Favorite leads after 5
    ud_covers = fav_f5_margin < -0.5  # Underdog leads after 5

    fav_cover_pct = fav_covers.mean()
    ud_cover_pct = ud_covers.mean()
    push_pct = (~fav_covers & ~ud_covers).mean()

    # Model accuracy
    fav_wp = np.where(fav_is_home, df["home_win_pct"], df["away_win_pct"])
    ud_wp = np.where(fav_is_home, df["away_win_pct"], df["home_win_pct"])
    pred_fav = fav_wp - ud_wp > 0.03
    accuracy = (pred_fav == fav_covers).mean()

    return {
        "market": "F5 Innings Run Line (-0.5/+0.5)",
        "games": int(len(df)),
        "fav_cover_pct": round(fav_cover_pct * 100, 1),
        "ud_cover_pct": round(ud_cover_pct * 100, 1),
        "push_pct": round(push_pct * 100, 1),
        "model_accuracy": round(accuracy * 100, 1),
    }


def main():
    log.info("=" * 60)
    log.info("COMPREHENSIVE MLB BACKTEST")
    log.info("=" * 60)

    # Step 1: Collect data
    log.info("\nStep 1: Collecting historical game data...")
    seasons = HISTORICAL_SEASONS
    games = collect_games_with_linescore(seasons)
    log.info(f"  Total games: {len(games)}")
    log.info(f"  Seasons: {sorted(games['season'].unique())}")

    # Step 2: Build rolling stats
    log.info("\nStep 2: Building rolling team statistics...")
    df = build_rolling_stats(games)

    # Filter out early-season games (need rolling data to stabilize)
    # Require at least 10 games per team
    df = df[
        (df["home_rpg"] != 4.5) & (df["away_rpg"] != 4.5)
    ].reset_index(drop=True)
    log.info(f"  Games with sufficient history: {len(df)}")

    # Step 3: Run all backtests
    log.info("\nStep 3: Running backtests...")
    results = {}

    results["ml_underdogs"] = backtest_ml_underdogs(df)
    results["full_game_totals"] = backtest_full_game_totals(df)
    results["run_line"] = backtest_run_line(df)
    results["team_totals"] = backtest_team_totals(df)
    results["f5_ml"] = backtest_f5_ml(df)
    results["f5_totals"] = backtest_f5_totals(df)
    results["f5_run_line"] = backtest_f5_run_line(df)

    # Per-season breakdown for key markets
    season_breakdown = {}
    for season in sorted(df["season"].unique()):
        sdf = df[df["season"] == season]
        season_breakdown[int(season)] = {
            "games": len(sdf),
            "ml_filtered": backtest_ml_underdogs(sdf),
            "full_game_totals": backtest_full_game_totals(sdf),
            "run_line": backtest_run_line(sdf),
        }

    # Step 4: Print report
    print(f"\n{'='*70}")
    print(f"  COMPREHENSIVE MLB BACKTEST RESULTS")
    print(f"  Seasons: {min(seasons)}-{max(seasons)} | Games: {len(df)}")
    print(f"{'='*70}")

    # ML Underdogs
    ml = results["ml_underdogs"]
    print(f"\n  1. ML UNDERDOGS")
    print(f"     Without filters: {ml['without_filters']['win_pct']}% "
          f"({ml['without_filters']['wins']}-{ml['without_filters']['losses']})")
    print(f"     With filters:    {ml['with_filters']['win_pct']}% "
          f"({ml['with_filters']['wins']}-{ml['with_filters']['losses']}, "
          f"{ml['with_filters']['total']} games)")

    # Full Game Totals
    fg = results["full_game_totals"]
    print(f"\n  2. FULL GAME TOTALS (O/U)")
    print(f"     Model accuracy: {fg['model_accuracy']}%")
    print(f"     Over hit rate:  {fg['over_pct']}%")
    print(f"     Games: {fg['games']} (excl {fg['pushes']} pushes)")

    # Run Line
    rl = results["run_line"]
    print(f"\n  3. RUN LINE (-1.5/+1.5)")
    print(f"     Fav covers -1.5: {rl['fav_cover_pct']}%")
    print(f"     Dog covers +1.5: {rl['ud_cover_pct']}%")
    print(f"     Model accuracy:  {rl['model_accuracy']}%")
    print(f"     Games: {rl['games']}")

    # Team Totals
    tt = results["team_totals"]
    print(f"\n  4. TEAM TOTALS (O/U)")
    print(f"     Home team accuracy: {tt['home_accuracy']}%")
    print(f"     Away team accuracy: {tt['away_accuracy']}%")
    print(f"     Combined accuracy:  {tt['combined_accuracy']}%")

    # F5 ML
    f5m = results["f5_ml"]
    print(f"\n  5. FIRST 5 INNINGS ML")
    print(f"     Model accuracy: {f5m['model_accuracy']}%")
    print(f"     Decided games:  {f5m['decided_games']} ({f5m['tie_pct']}% ties)")
    print(f"     Home win rate:  {f5m['home_win_pct']}%")

    # F5 Totals
    f5t = results["f5_totals"]
    print(f"\n  6. FIRST 5 INNINGS TOTALS (O/U)")
    print(f"     Model accuracy: {f5t['model_accuracy']}%")
    print(f"     Over hit rate:  {f5t['over_pct']}%")
    print(f"     Games: {f5t['games']}")

    # F5 Run Line
    f5r = results["f5_run_line"]
    print(f"\n  7. FIRST 5 INNINGS RUN LINE (-0.5/+0.5)")
    print(f"     Fav covers -0.5: {f5r['fav_cover_pct']}%")
    print(f"     Dog covers +0.5: {f5r['ud_cover_pct']}%")
    print(f"     Push rate:       {f5r['push_pct']}%")
    print(f"     Model accuracy:  {f5r['model_accuracy']}%")

    # Per-season breakdown
    print(f"\n  {'─'*60}")
    print(f"  PER-SEASON BREAKDOWN")
    print(f"  {'─'*60}")
    for season, sdata in season_breakdown.items():
        ml_f = sdata["ml_filtered"]["with_filters"]
        fg_s = sdata["full_game_totals"]
        rl_s = sdata["run_line"]
        print(f"\n  {season} ({sdata['games']} games):")
        print(f"    ML Dogs (filtered): {ml_f['win_pct']}% ({ml_f['wins']}-{ml_f['losses']})")
        print(f"    Full Game Totals:   {fg_s['model_accuracy']}%")
        print(f"    Run Line:           {rl_s['model_accuracy']}%")

    print(f"\n{'='*70}")

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "backtest_comprehensive.json", "w") as f:
        json.dump({"overall": results, "per_season": season_breakdown}, f, indent=2, default=str)
    log.info(f"Results saved to {OUTPUT_DIR / 'backtest_comprehensive.json'}")


if __name__ == "__main__":
    main()
