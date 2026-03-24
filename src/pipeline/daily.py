"""
Daily prediction pipeline.
This is the main entry point for the cron job.
Fetches today's games + odds, builds features, generates predictions.
"""

import json
import pandas as pd
from datetime import date, datetime

from config.settings import (
    OUTPUT_DIR, MIN_UNDERDOG_ODDS, MAX_UNDERDOG_ODDS,
    ODDS_API_KEY, RETRAIN_DAILY, ABBREV_TO_TEAM_ID, TEAM_ABBREVS,
)
from src.ingest.mlb_statsapi import (
    get_schedule, get_pitcher_season_stats, get_pitcher_game_log,
    get_team_batting_stats, get_team_pitching_stats, get_standings,
)
from src.ingest.odds_api import fetch_mlb_odds
from src.features.builder import build_feature_vector
from src.model.predict import generate_predictions
from src.model.registry import load_latest_model
from src.utils.odds_math import american_to_implied, remove_vig
from src.utils.logging import get_logger

log = get_logger(__name__)


def run_daily_pipeline(target_date: date = None) -> pd.DataFrame:
    """
    Run the full daily prediction pipeline.

    1. Fetch today's MLB schedule
    2. Fetch live odds
    3. Match games with odds
    4. Build features for qualifying underdog games
    5. Generate predictions
    6. Output results

    Returns:
        DataFrame of today's picks
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
        return pd.DataFrame()

    log.info(f"  Found {len(schedule)} games")

    # Step 2: Fetch live odds
    log.info("Step 2: Fetching live odds...")
    if ODDS_API_KEY:
        odds_data = fetch_mlb_odds()
    else:
        log.warning("No ODDS_API_KEY set. Using schedule without live odds.")
        odds_data = []

    # Step 3: Match games with odds and find qualifying underdogs
    log.info("Step 3: Matching games with odds...")
    matched_games = _match_games_with_odds(schedule, odds_data)
    qualifying = [g for g in matched_games if g.get("is_qualifying")]

    if not qualifying:
        log.info("No qualifying underdog games today.")
        _save_empty_output(target_date)
        return pd.DataFrame()

    log.info(f"  {len(qualifying)} qualifying underdog games")

    # Step 4: Build features
    log.info("Step 4: Building features for qualifying games...")
    standings = get_standings(season, as_of_date=target_date)

    games_with_features = []
    for game in qualifying:
        try:
            features = _build_game_features(game, season, standings)
            if features:
                games_with_features.append(features)
        except Exception as e:
            log.warning(f"Failed to build features for {game.get('home_team')} vs "
                       f"{game.get('away_team')}: {e}")

    if not games_with_features:
        log.info("Could not build features for any qualifying games.")
        _save_empty_output(target_date)
        return pd.DataFrame()

    # Step 5: Generate predictions
    log.info("Step 5: Generating predictions...")
    predictions = generate_predictions(games_with_features)

    # Step 6: Save output
    log.info("Step 6: Saving output...")
    _save_output(predictions, target_date)

    # Log summary
    recommended = predictions[predictions["recommended"]]
    log.info(f"\n{'='*50}")
    log.info(f"DAILY PICKS FOR {target_date}")
    log.info(f"Total qualifying games: {len(predictions)}")
    log.info(f"Recommended plays: {len(recommended)}")
    if not recommended.empty:
        for _, pick in recommended.iterrows():
            log.info(
                f"  >> {pick['underdog_team']} ({pick['underdog_odds']:+d}) | "
                f"Win Prob: {pick['model_win_prob']:.1%} | "
                f"Edge: {pick['edge_pct']} | {pick['confidence']}"
            )
    log.info(f"{'='*50}")

    return predictions


def _match_games_with_odds(schedule: list[dict], odds_data: list[dict]) -> list[dict]:
    """Match scheduled games with live odds data."""
    # Build lookup from odds data by team names
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
            })
        else:
            # Try fuzzy match by normalized names
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
                    })
                    found = True
                    break

            if not found:
                game["is_qualifying"] = False

        matched.append(game)

    return matched


def _normalize_team_pair(home: str, away: str):
    """Normalize team names to abbreviations."""
    h = TEAM_ABBREVS.get(home, "")
    a = TEAM_ABBREVS.get(away, "")
    if h and a:
        return (h, a)
    return None


def _build_game_features(game: dict, season: int, standings: dict) -> dict:
    """Build complete feature vector for a single game."""
    home = game["home_team"]
    away = game["away_team"]
    underdog_side = game.get("underdog", "away")
    ud_team = home if underdog_side == "home" else away
    fav_team = away if underdog_side == "home" else home

    ud_team_id = ABBREV_TO_TEAM_ID.get(ud_team)
    fav_team_id = ABBREV_TO_TEAM_ID.get(fav_team)

    # SP IDs
    if underdog_side == "home":
        ud_sp_id = game.get("home_sp_id")
        fav_sp_id = game.get("away_sp_id")
    else:
        ud_sp_id = game.get("away_sp_id")
        fav_sp_id = game.get("home_sp_id")

    # Fetch stats
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

    # Add game metadata
    features["game_date"] = game.get("game_date", "")
    features["home_team"] = home
    features["away_team"] = away
    features["underdog_team"] = ud_team
    features["home_sp_name"] = game.get("home_sp_name", "TBD")
    features["away_sp_name"] = game.get("away_sp_name", "TBD")

    return features


def _save_output(predictions: pd.DataFrame, target_date: date):
    """Save predictions to CSV and JSON."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    date_str = target_date.strftime("%Y-%m-%d")

    csv_path = OUTPUT_DIR / f"picks_{date_str}.csv"
    predictions.to_csv(csv_path, index=False)
    log.info(f"  Saved CSV: {csv_path}")

    json_path = OUTPUT_DIR / f"picks_{date_str}.json"
    predictions.to_json(json_path, orient="records", indent=2)
    log.info(f"  Saved JSON: {json_path}")


def _save_empty_output(target_date: date):
    """Save empty output when no picks are available."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    date_str = target_date.strftime("%Y-%m-%d")
    csv_path = OUTPUT_DIR / f"picks_{date_str}.csv"
    pd.DataFrame().to_csv(csv_path, index=False)
    json_path = OUTPUT_DIR / f"picks_{date_str}.json"
    with open(json_path, "w") as f:
        json.dump([], f)
