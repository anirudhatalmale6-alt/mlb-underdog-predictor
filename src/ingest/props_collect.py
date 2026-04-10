"""
Collect historical player prop data for backtesting.
Fetches pitcher game logs (for strikeouts) and batter game logs (for hits)
from the MLB Stats API.
"""

import time
import requests
import pandas as pd
from datetime import date
from typing import Optional

from src.utils.logging import get_logger

log = get_logger(__name__)

MLB_API = "https://statsapi.mlb.com/api/v1"
DELAY = 0.3  # Rate limiting


def get_season_starters(season: int) -> list[dict]:
    """
    Get all starting pitchers for a season by scanning the schedule.
    Returns list of {game_date, game_pk, home_team, away_team, home_sp_id, away_sp_id, ...}
    """
    log.info(f"Fetching {season} schedule for starter data...")
    url = f"{MLB_API}/schedule?season={season}&sportId=1&gameType=R&hydrate=probablePitcher,linescore"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()

    games = []
    for d in data.get("dates", []):
        game_date = d["date"]
        for g in d.get("games", []):
            if g.get("status", {}).get("abstractGameCode") != "F":
                continue

            home = g.get("teams", {}).get("home", {})
            away = g.get("teams", {}).get("away", {})
            home_team = home.get("team", {}).get("abbreviation", "")
            away_team = away.get("team", {}).get("abbreviation", "")

            home_sp = home.get("probablePitcher", {})
            away_sp = away.get("probablePitcher", {})

            # Get linescore for inning-level data
            linescore = g.get("linescore", {})
            home_score = linescore.get("teams", {}).get("home", {}).get("runs", 0)
            away_score = linescore.get("teams", {}).get("away", {}).get("runs", 0)

            games.append({
                "game_date": game_date,
                "game_pk": g.get("gamePk"),
                "home_team": home_team,
                "away_team": away_team,
                "home_sp_id": home_sp.get("id"),
                "away_sp_id": away_sp.get("id"),
                "home_sp_name": home_sp.get("fullName", ""),
                "away_sp_name": away_sp.get("fullName", ""),
                "home_score": home_score,
                "away_score": away_score,
                "total_runs": home_score + away_score,
            })

    log.info(f"  Found {len(games)} completed games in {season}")
    return games


def get_pitcher_game_log(pitcher_id: int, season: int) -> list[dict]:
    """Get a pitcher's game-by-game log for a season."""
    if not pitcher_id:
        return []

    url = f"{MLB_API}/people/{pitcher_id}/stats?stats=gameLog&season={season}&group=pitching"
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        data = r.json()
        time.sleep(DELAY)

        splits = data.get("stats", [{}])[0].get("splits", [])
        results = []
        for g in splits:
            s = g.get("stat", {})
            if s.get("gamesStarted", 0) == 0:
                continue  # Only starts
            results.append({
                "date": g.get("date"),
                "opponent": g.get("opponent", {}).get("name", ""),
                "opponent_id": g.get("opponent", {}).get("id"),
                "is_home": g.get("isHome", False),
                "ip": float(s.get("inningsPitched", 0)),
                "strikeouts": int(s.get("strikeOuts", 0)),
                "hits": int(s.get("hits", 0)),
                "walks": int(s.get("baseOnBalls", 0)),
                "earned_runs": int(s.get("earnedRuns", 0)),
                "pitches": int(s.get("numberOfPitches", 0)),
                "batters_faced": int(s.get("battersFaced", 0)),
                "home_runs": int(s.get("homeRuns", 0)),
            })
        return results
    except Exception as e:
        log.warning(f"Error fetching pitcher {pitcher_id} log: {e}")
        return []


def get_batter_game_log(batter_id: int, season: int) -> list[dict]:
    """Get a batter's game-by-game log for a season."""
    if not batter_id:
        return []

    url = f"{MLB_API}/people/{batter_id}/stats?stats=gameLog&season={season}&group=hitting"
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        data = r.json()
        time.sleep(DELAY)

        splits = data.get("stats", [{}])[0].get("splits", [])
        results = []
        for g in splits:
            s = g.get("stat", {})
            ab = int(s.get("atBats", 0))
            if ab == 0:
                continue  # Didn't bat
            results.append({
                "date": g.get("date"),
                "opponent": g.get("opponent", {}).get("name", ""),
                "opponent_id": g.get("opponent", {}).get("id"),
                "is_home": g.get("isHome", False),
                "at_bats": ab,
                "hits": int(s.get("hits", 0)),
                "doubles": int(s.get("doubles", 0)),
                "triples": int(s.get("triples", 0)),
                "home_runs": int(s.get("homeRuns", 0)),
                "strikeouts": int(s.get("strikeOuts", 0)),
                "walks": int(s.get("baseOnBalls", 0)),
                "total_bases": int(s.get("totalBases", 0)),
            })
        return results
    except Exception as e:
        log.warning(f"Error fetching batter {batter_id} log: {e}")
        return []


def get_team_roster_batters(team_id: int, season: int) -> list[dict]:
    """Get the primary lineup batters for a team in a season (top 9 by PA)."""
    url = f"{MLB_API}/teams/{team_id}/stats?stats=season&season={season}&group=hitting&gameType=R"
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        # Use the roster endpoint instead
        url2 = f"{MLB_API}/teams/{team_id}/roster?season={season}&rosterType=fullSeason"
        r2 = requests.get(url2, timeout=15)
        r2.raise_for_status()
        data = r2.json()
        time.sleep(DELAY)

        batters = []
        for p in data.get("roster", []):
            pos = p.get("position", {}).get("abbreviation", "")
            if pos == "P":
                continue
            batters.append({
                "player_id": p.get("person", {}).get("id"),
                "name": p.get("person", {}).get("fullName", ""),
                "position": pos,
            })
        return batters
    except Exception as e:
        log.warning(f"Error fetching roster for team {team_id}: {e}")
        return []


def get_team_strikeout_rate(team_id: int, season: int) -> float:
    """Get a team's batting strikeout rate for the season."""
    url = f"{MLB_API}/teams/{team_id}/stats?stats=season&season={season}&group=hitting&gameType=R"
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        data = r.json()
        time.sleep(DELAY)

        splits = data.get("stats", [{}])[0].get("splits", [])
        if splits:
            s = splits[0].get("stat", {})
            ab = int(s.get("atBats", 1))
            ks = int(s.get("strikeOuts", 0))
            return ks / max(ab, 1)
        return 0.22  # league average
    except Exception:
        return 0.22
