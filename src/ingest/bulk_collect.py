"""
Bulk data collection using MLB StatsAPI schedule endpoint.
Much faster than day-by-day collection — fetches an entire season in one call.
"""

import requests
import time
import pandas as pd
from datetime import date

from config.settings import MLB_API_BASE, TEAM_ID_TO_ABBREV, PROCESSED_DIR
from src.utils.logging import get_logger

log = get_logger(__name__)


def collect_season_bulk(season: int) -> pd.DataFrame:
    """
    Collect all regular-season game results for a season in bulk.
    Uses date range in a single API call instead of day-by-day.
    """
    from src.utils.dates import season_start, season_end

    start = season_start(season)
    end = season_end(season)

    log.info(f"Bulk collecting {season} season ({start} to {end})...")

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

            games.append({
                "game_id": game["gamePk"],
                "game_date": date_entry["date"],
                "home_team_id": home_id,
                "away_team_id": away_id,
                "home_team": TEAM_ID_TO_ABBREV.get(home_id, "UNK"),
                "away_team": TEAM_ID_TO_ABBREV.get(away_id, "UNK"),
                "home_score": home.get("score", 0),
                "away_score": away.get("score", 0),
                "home_win": 1 if home.get("score", 0) > away.get("score", 0) else 0,
            })

    df = pd.DataFrame(games)
    if not df.empty:
        df["game_date"] = pd.to_datetime(df["game_date"])
        df = df.drop_duplicates(subset=["game_id"])
    log.info(f"  Collected {len(df)} games for {season}")
    time.sleep(1)  # Be polite
    return df


def collect_multiple_seasons(seasons: list[int]) -> pd.DataFrame:
    """Collect multiple seasons and combine."""
    all_dfs = []
    for season in seasons:
        df = collect_season_bulk(season)
        if not df.empty:
            all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.sort_values("game_date").reset_index(drop=True)

    # Cache
    cache_path = PROCESSED_DIR / "historical_games.parquet"
    combined.to_parquet(cache_path, index=False)
    log.info(f"Cached {len(combined)} total games to {cache_path}")

    return combined
