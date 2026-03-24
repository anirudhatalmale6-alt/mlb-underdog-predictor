"""
Daily prediction pipeline.
This is the main entry point for the cron job.
Fetches today's games + odds, builds features, generates predictions.
Handles: moneyline underdogs, full-game totals, and F5 totals.
"""

import json
import pandas as pd
from datetime import date, datetime

from config.settings import (
    OUTPUT_DIR, MIN_UNDERDOG_ODDS, MAX_UNDERDOG_ODDS,
    ODDS_API_KEY, RETRAIN_DAILY, ABBREV_TO_TEAM_ID, TEAM_ABBREVS,
    PARK_FACTORS,
)
from src.ingest.mlb_statsapi import (
    get_schedule, get_pitcher_season_stats, get_pitcher_game_log,
    get_team_batting_stats, get_team_pitching_stats, get_standings,
)
from src.ingest.odds_api import fetch_mlb_odds
from src.features.builder import build_feature_vector
from src.model.predict import generate_predictions
from src.model.totals import predict_full_game_totals, predict_f5_totals
from src.model.registry import load_latest_model
from src.utils.odds_math import american_to_implied, remove_vig
from src.utils.logging import get_logger

log = get_logger(__name__)


def run_daily_pipeline(target_date: date = None) -> dict:
    """
    Run the full daily prediction pipeline.
    Returns dict with 'moneyline', 'full_game_total', 'f5_total' DataFrames.
    """
    if target_date is None:
        target_date = date.today()

    log.info(f"{'='*50}")
    log.info(f"Daily pipeline starting for {target_date}")
    log.info(f"{'='*50}")

    season = target_date.year

    # Step 1: Get today's schedule
    log.info("Step 1: Fetching today's schedule...")
    schedule = get_schedule(target_date)
    if not schedule:
        log.info("No games scheduled today.")
        return _empty_results()

    log.info(f"  Found {len(schedule)} games")

    # Step 2: Fetch live odds (now includes totals)
    log.info("Step 2: Fetching live odds...")
    if ODDS_API_KEY:
        odds_data = fetch_mlb_odds(include_totals=True)
    else:
        log.warning("No ODDS_API_KEY set.")
        odds_data = []

    # Step 3: Match games with odds
    log.info("Step 3: Matching games with odds...")
    matched_games = _match_games_with_odds(schedule, odds_data)

    # Step 4: Build features and standings
    log.info("Step 4: Fetching standings and building features...")
    standings = get_standings(season, as_of_date=target_date)

    results = {
        "moneyline": pd.DataFrame(),
        "full_game_total": pd.DataFrame(),
        "f5_total": pd.DataFrame(),
    }

    # ── Moneyline Underdog Picks ──
    qualifying_ml = [g for g in matched_games if g.get("is_qualifying")]
    if qualifying_ml:
        log.info(f"  {len(qualifying_ml)} qualifying underdog games")
        ml_features = []
        for game in qualifying_ml:
            try:
                features = _build_ml_features(game, season, standings)
                if features:
                    ml_features.append(features)
            except Exception as e:
                log.warning(f"Failed ML features for {game.get('home_team')} vs {game.get('away_team')}: {e}")

        if ml_features:
            results["moneyline"] = generate_predictions(ml_features)

    # ── Totals Picks (all games with total lines) ──
    games_with_totals = [g for g in matched_games if g.get("has_total")]
    if games_with_totals:
        log.info(f"  {len(games_with_totals)} games with totals lines")
        totals_features = []
        for game in games_with_totals:
            try:
                features = _build_totals_features_live(game, season, standings)
                if features:
                    totals_features.append(features)
            except Exception as e:
                log.warning(f"Failed totals features for {game.get('home_team')} vs {game.get('away_team')}: {e}")

        if totals_features:
            try:
                results["full_game_total"] = predict_full_game_totals(totals_features)
            except Exception as e:
                log.warning(f"Full game total prediction error: {e}")

            try:
                results["f5_total"] = predict_f5_totals(totals_features)
            except Exception as e:
                log.warning(f"F5 total prediction error: {e}")

    # Step 5: Save output
    log.info("Step 5: Saving output...")
    _save_output(results, target_date)

    # Log summary
    for bet_type, df in results.items():
        if not df.empty:
            rec = df[df["recommended"]].shape[0] if "recommended" in df.columns else 0
            log.info(f"  {bet_type}: {len(df)} picks, {rec} recommended")

    return results


def _empty_results():
    return {
        "moneyline": pd.DataFrame(),
        "full_game_total": pd.DataFrame(),
        "f5_total": pd.DataFrame(),
    }


def _match_games_with_odds(schedule: list[dict], odds_data: list[dict]) -> list[dict]:
    """Match scheduled games with live odds data."""
    odds_lookup = {}
    for odds in odds_data:
        key = _normalize_team_pair(odds.get("home_team", ""), odds.get("away_team", ""))
        if key:
            odds_lookup[key] = odds

    matched = []
    for game in schedule:
        home = game["home_team"]
        away = game["away_team"]
        key = (home, away)

        if key in odds_lookup:
            odds = odds_lookup[key]
            game.update({
                "home_consensus_odds": odds["home_consensus_odds"],
                "away_consensus_odds": odds["away_consensus_odds"],
                "underdog": odds["underdog"],
                "underdog_team": odds["underdog_team"],
                "underdog_odds": odds["underdog_odds"],
                "is_qualifying": odds["is_qualifying"],
                "market_implied_prob": odds["underdog_implied_prob"],
                "has_total": odds.get("has_total", False),
                "total_line": odds.get("total_line", 0),
                "over_odds": odds.get("over_odds", -110),
                "under_odds": odds.get("under_odds", -110),
            })
        else:
            # Try fuzzy match
            found = False
            for odds_key, odds in odds_lookup.items():
                odds_home_abbrev = TEAM_ABBREVS.get(odds.get("home_team", ""), "")
                odds_away_abbrev = TEAM_ABBREVS.get(odds.get("away_team", ""), "")
                if odds_home_abbrev == home and odds_away_abbrev == away:
                    game.update({
                        "home_consensus_odds": odds["home_consensus_odds"],
                        "away_consensus_odds": odds["away_consensus_odds"],
                        "underdog": odds["underdog"],
                        "underdog_team": home if odds["underdog"] == "home" else away,
                        "underdog_odds": odds["underdog_odds"],
                        "is_qualifying": odds["is_qualifying"],
                        "market_implied_prob": odds["underdog_implied_prob"],
                        "has_total": odds.get("has_total", False),
                        "total_line": odds.get("total_line", 0),
                        "over_odds": odds.get("over_odds", -110),
                        "under_odds": odds.get("under_odds", -110),
                    })
                    found = True
                    break

            if not found:
                game["is_qualifying"] = False
                game["has_total"] = False

        matched.append(game)

    return matched


def _normalize_team_pair(home: str, away: str):
    h = TEAM_ABBREVS.get(home, "")
    a = TEAM_ABBREVS.get(away, "")
    if h and a:
        return (h, a)
    return None


def _build_ml_features(game: dict, season: int, standings: dict) -> dict:
    """Build moneyline underdog features (existing logic)."""
    home = game["home_team"]
    away = game["away_team"]
    underdog_side = game.get("underdog", "away")
    ud_team = home if underdog_side == "home" else away
    fav_team = away if underdog_side == "home" else home

    ud_team_id = ABBREV_TO_TEAM_ID.get(ud_team)
    fav_team_id = ABBREV_TO_TEAM_ID.get(fav_team)

    if underdog_side == "home":
        ud_sp_id = game.get("home_sp_id")
        fav_sp_id = game.get("away_sp_id")
    else:
        ud_sp_id = game.get("away_sp_id")
        fav_sp_id = game.get("home_sp_id")

    ud_sp_stats = get_pitcher_season_stats(ud_sp_id, season)
    fav_sp_stats = get_pitcher_season_stats(fav_sp_id, season)
    ud_sp_logs = get_pitcher_game_log(ud_sp_id, season) if ud_sp_id else []
    fav_sp_logs = get_pitcher_game_log(fav_sp_id, season) if fav_sp_id else []

    ud_batting = get_team_batting_stats(ud_team_id, season) if ud_team_id else None
    fav_batting = get_team_batting_stats(fav_team_id, season) if fav_team_id else None
    ud_pitching = get_team_pitching_stats(ud_team_id, season) if ud_team_id else None
    fav_pitching = get_team_pitching_stats(fav_team_id, season) if fav_team_id else None

    ud_standings = standings.get(ud_team, {})
    fav_standings = standings.get(fav_team, {})

    features = build_feature_vector(
        underdog_sp_season=ud_sp_stats,
        favorite_sp_season=fav_sp_stats,
        underdog_batting=ud_batting,
        favorite_batting=fav_batting,
        underdog_bullpen_pitching=ud_pitching,
        favorite_bullpen_pitching=fav_pitching,
        underdog_sp_raw=ud_sp_stats,
        favorite_sp_raw=fav_sp_stats,
        underdog_standings=ud_standings,
        favorite_standings=fav_standings,
        underdog_sp_logs=ud_sp_logs,
        favorite_sp_logs=fav_sp_logs,
        home_team=home,
        underdog_side=underdog_side,
        underdog_odds=game.get("underdog_odds", 150),
        market_implied_prob=game.get("market_implied_prob", 0.4),
    )

    features["game_date"] = game.get("game_date", "")
    features["home_team"] = home
    features["away_team"] = away
    features["underdog_team"] = ud_team
    features["home_sp_name"] = game.get("home_sp_name", "TBD")
    features["away_sp_name"] = game.get("away_sp_name", "TBD")

    return features


def _build_totals_features_live(game: dict, season: int, standings: dict) -> dict:
    """Build totals features from live game data."""
    home = game["home_team"]
    away = game["away_team"]
    home_id = ABBREV_TO_TEAM_ID.get(home)
    away_id = ABBREV_TO_TEAM_ID.get(away)

    if not home_id or not away_id:
        return None

    # Get team batting and pitching stats
    home_batting = get_team_batting_stats(home_id, season)
    away_batting = get_team_batting_stats(away_id, season)
    home_pitching = get_team_pitching_stats(home_id, season)
    away_pitching = get_team_pitching_stats(away_id, season)
    home_stand = standings.get(home, {})
    away_stand = standings.get(away, {})

    # SP stats
    home_sp = get_pitcher_season_stats(game.get("home_sp_id"), season) or {}
    away_sp = get_pitcher_season_stats(game.get("away_sp_id"), season) or {}

    # Derive runs per game
    home_games = home_stand.get("wins", 0) + home_stand.get("losses", 0)
    away_games = away_stand.get("wins", 0) + away_stand.get("losses", 0)
    home_rpg = home_stand.get("runs_scored", 0) / max(home_games, 1)
    away_rpg = away_stand.get("runs_scored", 0) / max(away_games, 1)
    home_rapg = home_stand.get("runs_allowed", 0) / max(home_games, 1)
    away_rapg = away_stand.get("runs_allowed", 0) / max(away_games, 1)

    pf = PARK_FACTORS.get(home, 100) / 100.0

    home_sp_era = home_sp.get("era", 4.5)
    away_sp_era = away_sp.get("era", 4.5)

    features = {
        "home_rpg": home_rpg,
        "away_rpg": away_rpg,
        "home_ops": (home_batting or {}).get("ops", 0.720),
        "away_ops": (away_batting or {}).get("ops", 0.720),
        "home_iso": (home_batting or {}).get("slg", 0.400) - (home_batting or {}).get("avg", 0.250),
        "away_iso": (away_batting or {}).get("slg", 0.400) - (away_batting or {}).get("avg", 0.250),
        "home_bat_k_rate": 0.225,
        "away_bat_k_rate": 0.225,
        "home_bat_bb_rate": 0.085,
        "away_bat_bb_rate": 0.085,
        "home_rapg": home_rapg,
        "away_rapg": away_rapg,
        "home_era": (home_pitching or {}).get("era", 4.5),
        "away_era": (away_pitching or {}).get("era", 4.5),
        "home_whip": (home_pitching or {}).get("whip", 1.30),
        "away_whip": (away_pitching or {}).get("whip", 1.30),
        "home_sp_era": home_sp_era,
        "away_sp_era": away_sp_era,
        "home_sp_whip": home_sp.get("whip", 1.30),
        "away_sp_whip": away_sp.get("whip", 1.30),
        "home_sp_k_per_9": home_sp.get("strikeouts", 0) / max(home_sp.get("ip", 1), 1) * 9 if home_sp.get("ip", 0) > 0 else 8.5,
        "away_sp_k_per_9": away_sp.get("strikeouts", 0) / max(away_sp.get("ip", 1), 1) * 9 if away_sp.get("ip", 0) > 0 else 8.5,
        "home_sp_bb_per_9": home_sp.get("walks", 0) / max(home_sp.get("ip", 1), 1) * 9 if home_sp.get("ip", 0) > 0 else 3.2,
        "away_sp_bb_per_9": away_sp.get("walks", 0) / max(away_sp.get("ip", 1), 1) * 9 if away_sp.get("ip", 0) > 0 else 3.2,
        "home_sp_ip_per_start": home_sp.get("ip", 0) / max(home_sp.get("games_started", 1), 1),
        "away_sp_ip_per_start": away_sp.get("ip", 0) / max(away_sp.get("games_started", 1), 1),
        "combined_rpg": home_rpg + away_rpg,
        "combined_rapg": home_rapg + away_rapg,
        "combined_ops": (home_batting or {}).get("ops", 0.720) + (away_batting or {}).get("ops", 0.720),
        "park_factor": pf,
        "sp_era_combined": (home_sp_era + away_sp_era) / 2,
        "sp_whip_combined": (home_sp.get("whip", 1.30) + away_sp.get("whip", 1.30)) / 2,
        "home_recent_total_avg": home_rpg + home_rapg,
        "away_recent_total_avg": away_rpg + away_rapg,
        "home_bp_era": (home_pitching or {}).get("era", 4.5) * 1.05,
        "away_bp_era": (away_pitching or {}).get("era", 4.5) * 1.05,
        "home_win_pct": home_stand.get("pct", 0.5),
        "away_win_pct": away_stand.get("pct", 0.5),
        "home_run_diff_pg": (home_rpg - home_rapg),
        "away_run_diff_pg": (away_rpg - away_rapg),
        # F5 features (estimate from SP stats)
        "home_f5_rpg": home_rpg * 0.55,
        "away_f5_rpg": away_rpg * 0.55,
        "combined_f5_rpg": (home_rpg + away_rpg) * 0.55,
        "home_recent_f5_avg": (home_rpg + home_rapg) * 0.55,
        "away_recent_f5_avg": (away_rpg + away_rapg) * 0.55,
        # Game metadata
        "game_date": game.get("game_date", ""),
        "home_team": home,
        "away_team": away,
        "home_sp_name": game.get("home_sp_name", "TBD"),
        "away_sp_name": game.get("away_sp_name", "TBD"),
        "total_line": game.get("total_line", 0),
    }

    return features


def _save_output(results: dict, target_date: date):
    """Save all predictions to CSV and JSON."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    date_str = target_date.strftime("%Y-%m-%d")

    all_picks = []
    for bet_type, df in results.items():
        if not df.empty:
            all_picks.append(df)

    if all_picks:
        combined = pd.concat(all_picks, ignore_index=True)
        combined.to_csv(OUTPUT_DIR / f"picks_{date_str}.csv", index=False)
        combined.to_json(OUTPUT_DIR / f"picks_{date_str}.json", orient="records", indent=2)
        log.info(f"  Saved to picks_{date_str}.csv / .json")
    else:
        pd.DataFrame().to_csv(OUTPUT_DIR / f"picks_{date_str}.csv", index=False)
        with open(OUTPUT_DIR / f"picks_{date_str}.json", "w") as f:
            json.dump([], f)


def _save_empty_output(target_date: date):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    date_str = target_date.strftime("%Y-%m-%d")
    pd.DataFrame().to_csv(OUTPUT_DIR / f"picks_{date_str}.csv", index=False)
    with open(OUTPUT_DIR / f"picks_{date_str}.json", "w") as f:
        json.dump([], f)
