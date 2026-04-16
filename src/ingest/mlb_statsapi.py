"""
MLB StatsAPI data ingestion.
Uses the official MLB Stats API (statsapi.mlb.com) directly via requests.
No API key required.
"""

import requests
import time
from datetime import date, timedelta
from typing import Optional
import pandas as pd

from config.settings import MLB_API_BASE, MLB_SPORT_ID, TEAM_ID_TO_ABBREV
from src.utils.logging import get_logger

log = get_logger(__name__)

# Rate limiting: be polite
REQUEST_DELAY = 0.5


def _get(endpoint: str, params: dict = None) -> dict:
    """Make a GET request to the MLB Stats API."""
    url = f"{MLB_API_BASE}/{endpoint}"
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    time.sleep(REQUEST_DELAY)
    return resp.json()


# ── Schedule & Games ─────────────────────────────────────────────────────

def get_schedule(game_date: date) -> list[dict]:
    """
    Get the MLB schedule for a given date.
    Returns list of game dicts with teams, probable pitchers, venue, etc.
    """
    data = _get("schedule", {
        "sportId": MLB_SPORT_ID,
        "date": game_date.strftime("%Y-%m-%d"),
        "hydrate": "probablePitcher,team,venue",
    })

    games = []
    for date_entry in data.get("dates", []):
        for game in date_entry.get("games", []):
            if game.get("gameType") != "R":  # Regular season only
                continue

            home = game.get("teams", {}).get("home", {})
            away = game.get("teams", {}).get("away", {})
            home_team_id = home.get("team", {}).get("id")
            away_team_id = away.get("team", {}).get("id")

            game_info = {
                "game_id": game.get("gamePk"),
                "game_date": game_date.strftime("%Y-%m-%d"),
                "home_team_id": home_team_id,
                "away_team_id": away_team_id,
                "home_team": TEAM_ID_TO_ABBREV.get(home_team_id, "UNK"),
                "away_team": TEAM_ID_TO_ABBREV.get(away_team_id, "UNK"),
                "home_sp_id": home.get("probablePitcher", {}).get("id"),
                "away_sp_id": away.get("probablePitcher", {}).get("id"),
                "home_sp_name": home.get("probablePitcher", {}).get("fullName", "TBD"),
                "away_sp_name": away.get("probablePitcher", {}).get("fullName", "TBD"),
                "venue": game.get("venue", {}).get("name", ""),
                "status": game.get("status", {}).get("detailedState", ""),
            }
            games.append(game_info)

    log.info(f"Found {len(games)} regular-season games on {game_date}")
    return games


def get_team_recent_schedule(team_id: int, game_date: date, lookback_days: int = 5) -> list[dict]:
    """
    Get a team's recent schedule (last N days) to detect travel situations.
    Returns list of dicts with date, venue, home/away status, game time.
    """
    start = game_date - timedelta(days=lookback_days)
    end = game_date - timedelta(days=1)  # Exclude today

    data = _get("schedule", {
        "sportId": MLB_SPORT_ID,
        "teamId": team_id,
        "startDate": start.strftime("%Y-%m-%d"),
        "endDate": end.strftime("%Y-%m-%d"),
        "hydrate": "team,venue",
    })

    recent = []
    for date_entry in data.get("dates", []):
        for game in date_entry.get("games", []):
            if game.get("gameType") != "R":
                continue
            status = game.get("status", {}).get("detailedState", "")
            if status not in ("Final", "Completed Early", "Game Over"):
                continue

            home_team = game.get("teams", {}).get("home", {}).get("team", {})
            home_id = home_team.get("id")
            is_home = (home_id == team_id)
            venue = game.get("venue", {}).get("name", "")
            game_dt = game.get("gameDate", "")  # ISO format with time
            home_abbrev = TEAM_ID_TO_ABBREV.get(home_id, "")

            recent.append({
                "date": date_entry.get("date", ""),
                "is_home": is_home,
                "venue": venue,
                "home_team": home_abbrev,
                "game_datetime": game_dt,
            })

    return sorted(recent, key=lambda x: x["date"])


def get_game_results(game_date: date) -> list[dict]:
    """Get final scores for games on a date (for historical data)."""
    data = _get("schedule", {
        "sportId": MLB_SPORT_ID,
        "date": game_date.strftime("%Y-%m-%d"),
        "hydrate": "linescore",
    })

    results = []
    for date_entry in data.get("dates", []):
        for game in date_entry.get("games", []):
            if game.get("gameType") != "R":
                continue
            status = game.get("status", {}).get("detailedState", "")
            if status != "Final":
                continue

            home = game["teams"]["home"]
            away = game["teams"]["away"]
            home_id = home["team"]["id"]
            away_id = away["team"]["id"]

            results.append({
                "game_id": game["gamePk"],
                "game_date": game_date.strftime("%Y-%m-%d"),
                "home_team_id": home_id,
                "away_team_id": away_id,
                "home_team": TEAM_ID_TO_ABBREV.get(home_id, "UNK"),
                "away_team": TEAM_ID_TO_ABBREV.get(away_id, "UNK"),
                "home_score": home.get("score", 0),
                "away_score": away.get("score", 0),
                "home_win": 1 if home.get("score", 0) > away.get("score", 0) else 0,
            })

    return results


# ── Pitcher Stats ────────────────────────────────────────────────────────

def get_pitcher_season_stats(pitcher_id: int, season: int) -> Optional[dict]:
    """Get a pitcher's season stats."""
    if not pitcher_id:
        return None
    try:
        data = _get(f"people/{pitcher_id}/stats", {
            "stats": "season",
            "season": season,
            "group": "pitching",
        })
        splits = data.get("stats", [{}])[0].get("splits", [])
        if not splits:
            return None
        stats = splits[0].get("stat", {})
        return _parse_pitcher_stats(stats, pitcher_id)
    except Exception as e:
        log.warning(f"Failed to get stats for pitcher {pitcher_id}: {e}")
        return None


def get_pitcher_game_log(pitcher_id: int, season: int) -> list[dict]:
    """Get a pitcher's game-by-game log for a season."""
    if not pitcher_id:
        return []
    try:
        data = _get(f"people/{pitcher_id}/stats", {
            "stats": "gameLog",
            "season": season,
            "group": "pitching",
        })
        splits = data.get("stats", [{}])[0].get("splits", [])
        logs = []
        for s in splits:
            stat = s.get("stat", {})
            logs.append({
                "date": s.get("date", ""),
                "opponent": s.get("opponent", {}).get("id"),
                "is_home": s.get("isHome", False),
                **_parse_pitcher_stats(stat, pitcher_id),
            })
        return logs
    except Exception as e:
        log.warning(f"Failed to get game log for pitcher {pitcher_id}: {e}")
        return []


def _parse_pitcher_stats(stats: dict, pitcher_id: int) -> dict:
    """Parse raw pitcher stats dict into clean format."""
    ip_str = stats.get("inningsPitched", "0")
    try:
        ip = float(ip_str)
    except (ValueError, TypeError):
        ip = 0.0

    return {
        "pitcher_id": pitcher_id,
        "era": _safe_float(stats.get("era")),
        "whip": _safe_float(stats.get("whip")),
        "ip": ip,
        "strikeouts": _safe_int(stats.get("strikeOuts")),
        "walks": _safe_int(stats.get("baseOnBalls")),
        "hits": _safe_int(stats.get("hits")),
        "home_runs": _safe_int(stats.get("homeRuns")),
        "earned_runs": _safe_int(stats.get("earnedRuns")),
        "games_started": _safe_int(stats.get("gamesStarted")),
        "wins": _safe_int(stats.get("wins")),
        "losses": _safe_int(stats.get("losses")),
    }


# ── Team Stats ───────────────────────────────────────────────────────────

def get_team_batting_stats(team_id: int, season: int) -> Optional[dict]:
    """Get team batting stats for a season."""
    try:
        data = _get(f"teams/{team_id}/stats", {
            "stats": "season",
            "season": season,
            "group": "hitting",
        })
        splits = data.get("stats", [{}])[0].get("splits", [])
        if not splits:
            return None
        stats = splits[0].get("stat", {})
        return {
            "team_id": team_id,
            "avg": _safe_float(stats.get("avg")),
            "obp": _safe_float(stats.get("obp")),
            "slg": _safe_float(stats.get("slg")),
            "ops": _safe_float(stats.get("ops")),
            "runs": _safe_int(stats.get("runs")),
            "hits": _safe_int(stats.get("hits")),
            "home_runs": _safe_int(stats.get("homeRuns")),
            "strikeouts": _safe_int(stats.get("strikeOuts")),
            "walks": _safe_int(stats.get("baseOnBalls")),
            "stolen_bases": _safe_int(stats.get("stolenBases")),
            "games": _safe_int(stats.get("gamesPlayed")),
            "at_bats": _safe_int(stats.get("atBats")),
        }
    except Exception as e:
        log.warning(f"Failed to get batting stats for team {team_id}: {e}")
        return None


def get_team_pitching_stats(team_id: int, season: int) -> Optional[dict]:
    """Get team pitching stats (for bullpen approximation)."""
    try:
        data = _get(f"teams/{team_id}/stats", {
            "stats": "season",
            "season": season,
            "group": "pitching",
        })
        splits = data.get("stats", [{}])[0].get("splits", [])
        if not splits:
            return None
        stats = splits[0].get("stat", {})
        return {
            "team_id": team_id,
            "era": _safe_float(stats.get("era")),
            "whip": _safe_float(stats.get("whip")),
            "strikeouts": _safe_int(stats.get("strikeOuts")),
            "walks": _safe_int(stats.get("baseOnBalls")),
            "hits": _safe_int(stats.get("hits")),
            "home_runs": _safe_int(stats.get("homeRuns")),
            "ip": _safe_float(stats.get("inningsPitched", "0")),
            "saves": _safe_int(stats.get("saves")),
        }
    except Exception as e:
        log.warning(f"Failed to get pitching stats for team {team_id}: {e}")
        return None


# ── Standings ────────────────────────────────────────────────────────────

def get_standings(season: int, as_of_date: Optional[date] = None) -> dict:
    """
    Get team standings. Returns dict keyed by team abbreviation.
    """
    params = {"leagueId": "103,104", "season": season}
    if as_of_date:
        params["date"] = as_of_date.strftime("%Y-%m-%d")

    data = _get("standings", params)
    standings = {}

    for record in data.get("records", []):
        for team_record in record.get("teamRecords", []):
            team_id = team_record.get("team", {}).get("id")
            abbrev = TEAM_ID_TO_ABBREV.get(team_id)
            if not abbrev:
                continue

            streak = team_record.get("streak", {})
            standings[abbrev] = {
                "wins": team_record.get("wins", 0),
                "losses": team_record.get("losses", 0),
                "pct": _safe_float(team_record.get("winningPercentage")),
                "runs_scored": team_record.get("runsScored", 0),
                "runs_allowed": team_record.get("runsAllowed", 0),
                "streak_type": streak.get("streakType", ""),
                "streak_number": streak.get("streakNumber", 0),
                "last_ten_wins": team_record.get("records", {}).get(
                    "splitRecords", [{}]
                ),
                "home_wins": 0,
                "home_losses": 0,
                "away_wins": 0,
                "away_losses": 0,
            }

            # Extract home/away splits
            for split in team_record.get("records", {}).get("splitRecords", []):
                if split.get("type") == "home":
                    standings[abbrev]["home_wins"] = split.get("wins", 0)
                    standings[abbrev]["home_losses"] = split.get("losses", 0)
                elif split.get("type") == "away":
                    standings[abbrev]["away_wins"] = split.get("wins", 0)
                    standings[abbrev]["away_losses"] = split.get("losses", 0)
                elif split.get("type") == "lastTen":
                    standings[abbrev]["last_ten_wins"] = split.get("wins", 0)
                    standings[abbrev]["last_ten_losses"] = split.get("losses", 0)

    log.info(f"Got standings for {len(standings)} teams, season {season}")
    return standings


# ── Roster ───────────────────────────────────────────────────────────────

def get_roster(team_id: int, season: int) -> list[dict]:
    """Get team roster."""
    try:
        data = _get(f"teams/{team_id}/roster", {
            "season": season,
            "rosterType": "active",
        })
        return [
            {
                "id": p["person"]["id"],
                "name": p["person"]["fullName"],
                "position": p.get("position", {}).get("abbreviation", ""),
                "status": p.get("status", {}).get("code", ""),
            }
            for p in data.get("roster", [])
        ]
    except Exception as e:
        log.warning(f"Failed to get roster for team {team_id}: {e}")
        return []


# ── Historical Game Collection ───────────────────────────────────────────

def collect_season_results(season: int) -> pd.DataFrame:
    """
    Collect all regular-season game results for a given season.
    Returns a DataFrame with one row per game.
    """
    from src.utils.dates import season_start, season_end, date_range

    start = season_start(season)
    end = season_end(season)
    all_results = []

    log.info(f"Collecting game results for {season} season ({start} to {end})...")

    for d in date_range(start, end):
        try:
            results = get_game_results(d)
            all_results.extend(results)
        except Exception as e:
            log.warning(f"Failed to get results for {d}: {e}")
            continue

    df = pd.DataFrame(all_results)
    if not df.empty:
        df["game_date"] = pd.to_datetime(df["game_date"])
        df = df.drop_duplicates(subset=["game_id"])
        log.info(f"Collected {len(df)} games for {season}")
    return df


# ── Helpers ──────────────────────────────────────────────────────────────

def _safe_float(val, default: float = 0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _safe_int(val, default: int = 0) -> int:
    try:
        return int(val)
    except (TypeError, ValueError):
        return default
