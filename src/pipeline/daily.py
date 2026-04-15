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
from src.ingest.odds_api import fetch_mlb_odds, fetch_mlb_event_markets
from src.features.builder import build_feature_vector
from src.features.props import build_pitcher_k_features, build_batter_hits_features
from src.model.predict import generate_predictions
from src.model.totals import predict_full_game_totals, predict_f5_totals
from src.model.props import predict_pitcher_k_props, predict_batter_hits_props
from src.model.first_inning import predict_first_inning_ml, predict_first_inning_total
from src.features.first_inning import FIRST_INNING_FEATURES
from src.model.registry import load_latest_model
from src.ingest.props_collect import get_pitcher_game_log as get_pitcher_log_raw, get_team_strikeout_rate, get_batter_game_log as get_batter_log_raw
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
        "i1_ml": [],
        "i1_total": [],
        "pitcher_k": pd.DataFrame(),
        "batter_hits": pd.DataFrame(),
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

    # ── Fetch all event-level markets in one batch (props + 1st inning) ──
    # Only fetch for games with known starting pitchers to save API quota
    log.info("Step 4b: Fetching event markets (props + 1st inning)...")
    event_ids = [
        g.get("event_id") for g in matched_games
        if g.get("event_id") and g.get("home_sp_id")
    ]
    all_event_markets = {}
    if event_ids:
        try:
            all_event_markets = fetch_mlb_event_markets(event_ids)
        except Exception as e:
            log.warning(f"Event markets fetch error: {e}")

    # ── Player Props (using pre-fetched data) ──
    try:
        props_by_event = {
            eid: {"pitcher_k": d.get("pitcher_k", []), "batter_hits": d.get("batter_hits", [])}
            for eid, d in all_event_markets.items()
        }
        props_data = _build_props_features(schedule, matched_games, season, props_by_event)
        if props_data.get("pitcher_k"):
            results["pitcher_k"] = predict_pitcher_k_props(props_data["pitcher_k"])
        if props_data.get("batter_hits"):
            results["batter_hits"] = predict_batter_hits_props(props_data["batter_hits"])
    except Exception as e:
        log.warning(f"Player props error: {e}")

    # ── 1st Inning Markets (using pre-fetched data) ──
    try:
        i1_odds = {
            eid: {k: v for k, v in d.items() if k.startswith("i1_")}
            for eid, d in all_event_markets.items()
            if d.get("i1_ml") or d.get("i1_total")
        }
        i1_data = _build_first_inning_features(matched_games, season, standings, i1_odds)
        if i1_data:
            results["i1_ml"] = predict_first_inning_ml(i1_data)
            results["i1_total"] = predict_first_inning_total(i1_data)
    except Exception as e:
        log.warning(f"1st inning error: {e}")

    # Step 5: Save output
    log.info("Step 5: Saving output...")
    _save_output(results, target_date)

    # Log summary
    for bet_type, data in results.items():
        if isinstance(data, list):
            if data:
                rec = sum(1 for p in data if p.get("recommended"))
                log.info(f"  {bet_type}: {len(data)} picks, {rec} recommended")
        elif isinstance(data, pd.DataFrame) and not data.empty:
            rec = data[data["recommended"]].shape[0] if "recommended" in data.columns else 0
            log.info(f"  {bet_type}: {len(data)} picks, {rec} recommended")

    return results


def _empty_results():
    return {
        "moneyline": pd.DataFrame(),
        "full_game_total": pd.DataFrame(),
        "f5_total": pd.DataFrame(),
        "i1_ml": [],
        "i1_total": [],
        "pitcher_k": pd.DataFrame(),
        "batter_hits": pd.DataFrame(),
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
                "event_id": odds.get("event_id", ""),
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
                        "event_id": odds.get("event_id", ""),
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


def _build_props_features(schedule: list, matched_games: list, season: int, props_by_event: dict = None) -> dict:
    """
    Build features for player props from live data + game logs.
    Uses pre-fetched props_by_event if provided, otherwise returns empty.
    """
    if not props_by_event:
        return {"pitcher_k": [], "batter_hits": []}

    pitcher_k_features = []
    batter_hits_features = []

    for game in matched_games:
        eid = game.get("event_id", "")
        home = game.get("home_team", "")
        away = game.get("away_team", "")
        game_date = game.get("game_date", "")

        props = props_by_event.get(eid, {})

        # ── Pitcher K props ──
        for sp_side, opp_side in [("home", "away"), ("away", "home")]:
            sp_id = game.get(f"{sp_side}_sp_id")
            sp_name = game.get(f"{sp_side}_sp_name", "TBD")
            opp_team = game.get(f"{opp_side}_team", "")
            opp_team_id = ABBREV_TO_TEAM_ID.get(opp_team)

            if not sp_id:
                continue

            # Get pitcher's game log for features (combine with prior season if needed)
            pitcher_log = get_pitcher_log_raw(sp_id, season)
            if len(pitcher_log) < 5:
                prior_log = get_pitcher_log_raw(sp_id, season - 1)
                pitcher_log = prior_log + pitcher_log
                if len(pitcher_log) < 5:
                    continue

            opp_k_rate = get_team_strikeout_rate(opp_team_id, season) if opp_team_id else 0.22

            features = build_pitcher_k_features(
                pitcher_log, len(pitcher_log),  # Use all data (no look-ahead since it's live)
                opp_k_rate=opp_k_rate,
                is_home=(sp_side == "home"),
                min_starts=5,
            )
            if features is None:
                continue

            # Find matching prop line
            k_line = 0
            k_over_odds = -110
            k_under_odds = -110
            for prop in props.get("pitcher_k", []):
                if _name_match(prop["player"], sp_name):
                    k_line = prop["line"]
                    k_over_odds = prop["over_odds"]
                    k_under_odds = prop["under_odds"]
                    break

            if k_line == 0:
                # Use model estimate as proxy
                k_line = round(features.get("k_per_start_avg", 5.5) - 0.5, 1)

            features.update({
                "game_date": game_date,
                "home_team": home,
                "away_team": away,
                "pitcher_name": sp_name,
                "pitcher_id": sp_id,
                "k_line": k_line,
                "k_over_odds": k_over_odds,
                "k_under_odds": k_under_odds,
            })
            pitcher_k_features.append(features)

        # ── Batter Hits props ──
        for prop in props.get("batter_hits", []):
            player_name = prop["player"]
            # Build features for this batter
            # We need to find the batter's ID - search through roster
            batter_features = _build_batter_features_from_prop(
                player_name, prop, game, season
            )
            if batter_features:
                batter_hits_features.append(batter_features)

    log.info(f"  Props: {len(pitcher_k_features)} pitcher K, {len(batter_hits_features)} batter hits")
    return {"pitcher_k": pitcher_k_features, "batter_hits": batter_hits_features}


def _build_batter_features_from_prop(player_name: str, prop: dict, game: dict, season: int) -> dict:
    """Build batter hit features for a specific prop."""
    import requests

    # Search for player by name
    try:
        r = requests.get(
            f"{MLB_API_BASE}/people/search?names={player_name}&sportId=1",
            timeout=10,
        )
        if r.status_code != 200:
            return None
        people = r.json().get("people", [])
        if not people:
            return None
        batter_id = people[0]["id"]
    except Exception:
        return None

    blog = get_batter_log_raw(batter_id, season)
    if len(blog) < 15:
        prior_blog = get_batter_log_raw(batter_id, season - 1)
        blog = prior_blog + blog
        if len(blog) < 15:
            return None

    features = build_batter_hits_features(
        blog, len(blog),
        opp_sp_stats=None,
        is_home=False,  # Approximate
        min_games=15,
    )
    if features is None:
        return None

    features.update({
        "game_date": game.get("game_date", ""),
        "home_team": game.get("home_team", ""),
        "away_team": game.get("away_team", ""),
        "batter_name": player_name,
        "batter_id": batter_id,
        "hits_line": prop["line"],
        "hits_over_odds": prop["over_odds"],
        "hits_under_odds": prop["under_odds"],
    })
    return features


def _name_match(prop_name: str, sp_name: str) -> bool:
    """Fuzzy match player names between prop API and schedule."""
    if not prop_name or not sp_name:
        return False
    # Normalize
    pn = prop_name.lower().strip()
    sn = sp_name.lower().strip()
    if pn == sn:
        return True
    # Last name match
    p_last = pn.split()[-1] if pn else ""
    s_last = sn.split()[-1] if sn else ""
    return p_last == s_last and len(p_last) > 2


MLB_API_BASE = "https://statsapi.mlb.com/api/v1"


def _build_first_inning_features(matched_games: list, season: int, standings: dict, i1_odds: dict = None) -> list[dict]:
    """Build features for 1st inning predictions using live SP data."""
    if i1_odds is None:
        i1_odds = {}

    features_list = []
    for game in matched_games:
        home = game.get("home_team", "")
        away = game.get("away_team", "")
        eid = game.get("event_id", "")

        # Get SP stats for both teams
        home_sp_id = game.get("home_sp_id")
        away_sp_id = game.get("away_sp_id")

        home_sp_stats = _get_sp_stats_for_i1(home_sp_id, season) if home_sp_id else {}
        away_sp_stats = _get_sp_stats_for_i1(away_sp_id, season) if away_sp_id else {}

        # Get team rolling stats from standings
        home_stand = standings.get(home, {})
        away_stand = standings.get(away, {})

        home_rpg = home_stand.get("rpg", 4.5)
        away_rpg = away_stand.get("rpg", 4.5)
        home_rapg = home_stand.get("rapg", 4.5)
        away_rapg = away_stand.get("rapg", 4.5)
        home_win_pct = home_stand.get("win_pct", 0.5)
        away_win_pct = away_stand.get("win_pct", 0.5)

        pf = PARK_FACTORS.get(home, 100) / 100.0

        features = {
            "game_date": game.get("game_date", ""),
            "home_team": home,
            "away_team": away,
            "event_id": eid,
            "home_sp_name": game.get("home_sp_name", "TBD"),
            "away_sp_name": game.get("away_sp_name", "TBD"),
            # SP stats
            "home_sp_era": home_sp_stats.get("era", home_rapg),
            "away_sp_era": away_sp_stats.get("era", away_rapg),
            "home_sp_whip": home_sp_stats.get("whip", 1.30),
            "away_sp_whip": away_sp_stats.get("whip", 1.30),
            "home_sp_k_per_9": home_sp_stats.get("k_per_9", 8.5),
            "away_sp_k_per_9": away_sp_stats.get("k_per_9", 8.5),
            "home_sp_bb_per_9": home_sp_stats.get("bb_per_9", 3.2),
            "away_sp_bb_per_9": away_sp_stats.get("bb_per_9", 3.2),
            "home_sp_hr_per_9": home_sp_stats.get("hr_per_9", 1.2),
            "away_sp_hr_per_9": away_sp_stats.get("hr_per_9", 1.2),
            # 1st inning specific (approximate from team+SP data)
            "home_1st_inn_rpg": home_rpg / 9.0,
            "away_1st_inn_rpg": away_rpg / 9.0,
            "home_1st_inn_rapg": home_rapg / 9.0,
            "away_1st_inn_rapg": away_rapg / 9.0,
            "home_1st_inn_scored_pct": min(0.35, home_rpg / 15.0),
            "away_1st_inn_scored_pct": min(0.35, away_rpg / 15.0),
            "home_1st_inn_allowed_pct": min(0.35, home_rapg / 15.0),
            "away_1st_inn_allowed_pct": min(0.35, away_rapg / 15.0),
            # Team offensive
            "home_rpg": home_rpg,
            "away_rpg": away_rpg,
            "home_ops": 0.720 + (home_rpg - 4.5) * 0.03,
            "away_ops": 0.720 + (away_rpg - 4.5) * 0.03,
            # Combined
            "combined_1st_inn_rpg": (home_rpg + away_rpg) / 9.0,
            "combined_1st_inn_score_pct": min(0.50, (home_rpg + away_rpg) / 30.0),
            "park_factor": pf,
            "sp_era_combined": (home_sp_stats.get("era", 4.5) + away_sp_stats.get("era", 4.5)) / 2,
            "sp_whip_combined": (home_sp_stats.get("whip", 1.30) + away_sp_stats.get("whip", 1.30)) / 2,
            # Momentum
            "home_win_pct": home_win_pct,
            "away_win_pct": away_win_pct,
        }

        # Add 1st inning odds if available
        eid_odds = i1_odds.get(eid, {})
        if "ml" in eid_odds:
            features["i1_home_odds"] = eid_odds["ml"]["home_odds"]
            features["i1_away_odds"] = eid_odds["ml"]["away_odds"]
        if "total" in eid_odds:
            features["i1_over_odds"] = eid_odds["total"]["over_odds"]
            features["i1_under_odds"] = eid_odds["total"]["under_odds"]

        features_list.append(features)

    log.info(f"  1st inning: {len(features_list)} games with features, {len(i1_odds)} with odds")
    return features_list


def _get_sp_stats_for_i1(sp_id: int, season: int) -> dict:
    """Get SP stats relevant to 1st inning prediction."""
    try:
        stats = get_pitcher_season_stats(sp_id, season)
        if not stats:
            # Try prior season
            stats = get_pitcher_season_stats(sp_id, season - 1)
        if not stats:
            return {}

        ip = stats.get("inningsPitched", "0")
        ip = float(ip) if ip else 0
        era = float(stats.get("era", "4.50"))
        whip = float(stats.get("whip", "1.30"))
        ks = int(stats.get("strikeOuts", 0))
        bbs = int(stats.get("baseOnBalls", 0))
        hrs = int(stats.get("homeRuns", 0))

        k_per_9 = (ks / ip * 9) if ip > 0 else 8.5
        bb_per_9 = (bbs / ip * 9) if ip > 0 else 3.2
        hr_per_9 = (hrs / ip * 9) if ip > 0 else 1.2

        return {
            "era": era,
            "whip": whip,
            "k_per_9": k_per_9,
            "bb_per_9": bb_per_9,
            "hr_per_9": hr_per_9,
        }
    except Exception:
        return {}


def _save_output(results: dict, target_date: date):
    """Save all predictions to CSV and JSON."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    date_str = target_date.strftime("%Y-%m-%d")

    all_picks = []
    for bet_type, data in results.items():
        if isinstance(data, list) and data:
            all_picks.append(pd.DataFrame(data))
        elif isinstance(data, pd.DataFrame) and not data.empty:
            all_picks.append(data)

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
