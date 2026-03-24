"""
Historical data collection and Elo-based synthetic odds generation.
For backtesting, we need game results + odds. Since free historical odds
data is limited, we build an Elo rating system to generate synthetic
moneyline odds for seasons where real odds aren't available.
"""

import json
import pandas as pd
import numpy as np
from datetime import date
from pathlib import Path

from config.settings import PROCESSED_DIR, RAW_DIR, HISTORICAL_SEASONS, TEAM_ID_TO_ABBREV
from src.utils.odds_math import elo_to_implied, implied_to_american
from src.utils.logging import get_logger

log = get_logger(__name__)

# Elo parameters
ELO_INITIAL = 1500
ELO_K = 6          # K-factor per game
ELO_HOME_ADV = 24  # Home advantage in Elo points
ELO_REGRESS = 0.6  # Between-season regression factor (toward mean)


def build_historical_dataset(seasons: list[int] = None) -> pd.DataFrame:
    """
    Build the master historical dataset for backtesting.
    Uses MLB StatsAPI to collect game results, then generates Elo-based
    synthetic odds for each game.

    Returns DataFrame with columns:
        game_id, game_date, home_team, away_team, home_score, away_score,
        home_win, home_elo, away_elo, home_odds, away_odds, underdog,
        underdog_odds, underdog_won
    """
    if seasons is None:
        seasons = HISTORICAL_SEASONS

    cache_path = PROCESSED_DIR / "historical_games.parquet"
    if cache_path.exists():
        log.info("Loading cached historical games...")
        return pd.read_parquet(cache_path)

    from src.ingest.mlb_statsapi import collect_season_results

    all_games = []
    for season in seasons:
        df = collect_season_results(season)
        if not df.empty:
            all_games.append(df)

    if not all_games:
        log.error("No historical games collected!")
        return pd.DataFrame()

    games = pd.concat(all_games, ignore_index=True)
    games = games.sort_values("game_date").reset_index(drop=True)

    # Generate Elo ratings and synthetic odds
    games = _compute_elo_and_odds(games)

    # Save cache
    games.to_parquet(cache_path, index=False)
    log.info(f"Built historical dataset: {len(games)} games across {len(seasons)} seasons")
    return games


def _compute_elo_and_odds(games: pd.DataFrame) -> pd.DataFrame:
    """
    Walk through games chronologically, maintain Elo ratings,
    and generate synthetic American odds for each game.
    """
    elo = {}  # team -> current elo
    home_elos = []
    away_elos = []
    home_odds = []
    away_odds = []
    current_season = None

    for _, row in games.iterrows():
        game_season = row["game_date"].year if hasattr(row["game_date"], "year") else pd.Timestamp(row["game_date"]).year
        home = row["home_team"]
        away = row["away_team"]

        # Season regression
        if game_season != current_season:
            if current_season is not None:
                for team in elo:
                    elo[team] = ELO_INITIAL + ELO_REGRESS * (elo[team] - ELO_INITIAL)
            current_season = game_season

        # Initialize teams if new
        if home not in elo:
            elo[home] = ELO_INITIAL
        if away not in elo:
            elo[away] = ELO_INITIAL

        # Pre-game Elos (what we'd know before the game)
        home_pre = elo[home]
        away_pre = elo[away]
        home_elos.append(home_pre)
        away_elos.append(away_pre)

        # Synthetic odds from Elo (with home advantage)
        elo_diff = home_pre - away_pre + ELO_HOME_ADV
        home_prob = elo_to_implied(elo_diff)
        away_prob = 1 - home_prob

        # Clamp to avoid extreme odds
        home_prob = np.clip(home_prob, 0.1, 0.9)
        away_prob = 1 - home_prob

        try:
            h_odds = implied_to_american(home_prob)
            a_odds = implied_to_american(away_prob)
        except ValueError:
            h_odds = -150
            a_odds = 130

        home_odds.append(h_odds)
        away_odds.append(a_odds)

        # Update Elo based on result
        home_score = row["home_score"]
        away_score = row["away_score"]
        actual = 1.0 if home_score > away_score else 0.0
        expected = elo_to_implied(home_pre - away_pre + ELO_HOME_ADV)

        # Margin-of-victory multiplier
        mov = abs(home_score - away_score)
        mov_mult = np.log(max(mov, 1) + 1) * (2.2 / (
            (home_pre - away_pre if actual == 1 else away_pre - home_pre) * 0.001 + 2.2
        ))

        elo[home] += ELO_K * mov_mult * (actual - expected)
        elo[away] += ELO_K * mov_mult * (expected - actual)

    games["home_elo"] = home_elos
    games["away_elo"] = away_elos
    games["home_odds"] = home_odds
    games["away_odds"] = away_odds

    # Determine underdog
    games["underdog"] = np.where(games["home_odds"] > 0, "home", "away")
    games["underdog_odds"] = np.where(
        games["underdog"] == "home",
        games["home_odds"],
        games["away_odds"]
    )
    games["underdog_team"] = np.where(
        games["underdog"] == "home",
        games["home_team"],
        games["away_team"]
    )
    games["underdog_won"] = np.where(
        games["underdog"] == "home",
        games["home_win"],
        1 - games["home_win"]
    )

    return games


def load_historical_dataset() -> pd.DataFrame:
    """Load the cached historical dataset."""
    cache_path = PROCESSED_DIR / "historical_games.parquet"
    if not cache_path.exists():
        log.warning("No cached historical data found. Run build_historical_dataset() first.")
        return pd.DataFrame()
    return pd.read_parquet(cache_path)
