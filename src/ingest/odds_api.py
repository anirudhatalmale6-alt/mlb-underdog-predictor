"""
The Odds API data ingestion.
Fetches live MLB moneyline odds from multiple bookmakers.
"""

import json
import requests
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from config.settings import (
    ODDS_API_KEY, ODDS_API_BASE, ODDS_REGIONS,
    ODDS_MARKETS, ODDS_FORMAT, RAW_DIR,
    MIN_UNDERDOG_ODDS, MAX_UNDERDOG_ODDS,
    TEAM_ABBREVS,
)
from src.utils.odds_math import american_to_implied, remove_vig, is_qualifying_underdog
from src.utils.logging import get_logger

log = get_logger(__name__)


def fetch_mlb_odds(include_totals: bool = True) -> list[dict]:
    """
    Fetch current MLB odds from The Odds API.
    Returns list of game dicts with moneyline + totals odds.
    """
    if not ODDS_API_KEY:
        log.error("ODDS_API_KEY not set. Cannot fetch live odds.")
        return []

    markets = "h2h,totals" if include_totals else ODDS_MARKETS

    url = f"{ODDS_API_BASE}/sports/baseball_mlb/odds/"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": ODDS_REGIONS,
        "markets": markets,
        "oddsFormat": ODDS_FORMAT,
    }

    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()

    # Log remaining quota
    remaining = resp.headers.get("x-requests-remaining", "?")
    used = resp.headers.get("x-requests-used", "?")
    log.info(f"Odds API quota: {remaining} remaining, {used} used")

    data = resp.json()

    # Save raw snapshot
    snapshot_dir = RAW_DIR / "odds_api"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(snapshot_dir / f"odds_{ts}.json", "w") as f:
        json.dump(data, f, indent=2)

    return _parse_odds_response(data)


def _parse_odds_response(data: list[dict]) -> list[dict]:
    """Parse The Odds API response into clean game records with moneyline + totals."""
    games = []
    for event in data:
        home_team = event.get("home_team", "")
        away_team = event.get("away_team", "")
        commence = event.get("commence_time", "")

        # Collect odds from all bookmakers
        home_odds_list = []
        away_odds_list = []
        total_points_list = []
        over_odds_list = []
        under_odds_list = []

        for bookmaker in event.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                key = market.get("key")
                outcomes = {o["name"]: o for o in market.get("outcomes", [])}

                if key == "h2h":
                    if home_team in outcomes and away_team in outcomes:
                        home_odds_list.append(outcomes[home_team]["price"])
                        away_odds_list.append(outcomes[away_team]["price"])

                elif key == "totals":
                    if "Over" in outcomes and "Under" in outcomes:
                        total_points_list.append(outcomes["Over"].get("point", 0))
                        over_odds_list.append(outcomes["Over"]["price"])
                        under_odds_list.append(outcomes["Under"]["price"])

        if not home_odds_list or not away_odds_list:
            continue

        # Consensus moneyline odds (median across books)
        home_odds_list.sort()
        away_odds_list.sort()
        mid = len(home_odds_list) // 2
        home_consensus = home_odds_list[mid]
        away_consensus = away_odds_list[mid]

        # Determine underdog
        if home_consensus > 0 and away_consensus < 0:
            underdog = "home"
            underdog_odds = home_consensus
            favorite_odds = away_consensus
        elif away_consensus > 0 and home_consensus < 0:
            underdog = "away"
            underdog_odds = away_consensus
            favorite_odds = home_consensus
        else:
            if home_consensus >= away_consensus:
                underdog = "home"
                underdog_odds = home_consensus
                favorite_odds = away_consensus
            else:
                underdog = "away"
                underdog_odds = away_consensus
                favorite_odds = home_consensus

        # True probabilities (vig removed)
        home_prob, away_prob = remove_vig(home_consensus, away_consensus)
        underdog_prob = home_prob if underdog == "home" else away_prob

        game = {
            "event_id": event.get("id", ""),
            "commence_time": commence,
            "home_team": home_team,
            "away_team": away_team,
            "home_consensus_odds": home_consensus,
            "away_consensus_odds": away_consensus,
            "underdog": underdog,
            "underdog_team": home_team if underdog == "home" else away_team,
            "underdog_odds": underdog_odds,
            "favorite_odds": favorite_odds,
            "underdog_implied_prob": underdog_prob,
            "is_qualifying": is_qualifying_underdog(
                underdog_odds, MIN_UNDERDOG_ODDS, MAX_UNDERDOG_ODDS
            ),
            "num_bookmakers": len(home_odds_list),
            "odds_spread": max(home_odds_list) - min(home_odds_list),
        }

        # Totals consensus
        if total_points_list and over_odds_list:
            total_points_list.sort()
            over_odds_list.sort()
            under_odds_list.sort()
            tmid = len(total_points_list) // 2
            game["total_line"] = total_points_list[tmid]
            game["over_odds"] = over_odds_list[tmid]
            game["under_odds"] = under_odds_list[tmid]
            game["has_total"] = True
        else:
            game["has_total"] = False

        games.append(game)

    qualifying = sum(1 for g in games if g["is_qualifying"])
    with_totals = sum(1 for g in games if g.get("has_total"))
    log.info(f"Parsed {len(games)} games, {qualifying} qualifying underdogs, {with_totals} with totals")
    return games


def load_latest_snapshot() -> Optional[list[dict]]:
    """Load the most recent odds snapshot from disk."""
    snapshot_dir = RAW_DIR / "odds_api"
    if not snapshot_dir.exists():
        return None
    files = sorted(snapshot_dir.glob("odds_*.json"), reverse=True)
    if not files:
        return None
    with open(files[0]) as f:
        data = json.load(f)
    return _parse_odds_response(data)
