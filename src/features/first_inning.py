"""
Feature engineering for 1st inning models.
Builds features for 1st inning moneyline and 1st inning totals (over/under 0.5).
Key insight: 1st inning is heavily dependent on starting pitchers and leadoff hitters.
"""

import numpy as np
import pandas as pd

from config.settings import PARK_FACTORS
from src.utils.logging import get_logger

log = get_logger(__name__)

# Features for 1st inning models
FIRST_INNING_FEATURES = [
    # SP quality indicators (most predictive for 1st inning)
    "home_sp_era", "away_sp_era",
    "home_sp_whip", "away_sp_whip",
    "home_sp_k_per_9", "away_sp_k_per_9",
    "home_sp_bb_per_9", "away_sp_bb_per_9",
    "home_sp_hr_per_9", "away_sp_hr_per_9",
    # 1st inning specific rolling stats
    "home_1st_inn_rpg", "away_1st_inn_rpg",       # Avg runs scored in 1st inning
    "home_1st_inn_rapg", "away_1st_inn_rapg",      # Avg runs allowed in 1st inning
    "home_1st_inn_scored_pct", "away_1st_inn_scored_pct",  # % of games team scores in 1st
    "home_1st_inn_allowed_pct", "away_1st_inn_allowed_pct",  # % of games allowing runs in 1st
    # Team offensive tendency
    "home_rpg", "away_rpg",
    "home_ops", "away_ops",
    # Combined/derived
    "combined_1st_inn_rpg",         # Both teams' 1st inning scoring combined
    "combined_1st_inn_score_pct",   # Avg pct of scoring in 1st
    "park_factor",
    "sp_era_combined",
    "sp_whip_combined",
    # Situational
    "home_win_pct", "away_win_pct",
]


def build_first_inning_features_historical(games_df: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    """
    Build 1st inning features from historical game data.
    Requires: home_score, away_score, i1_home_runs, i1_away_runs, home_team, away_team.
    """
    df = games_df.sort_values("game_date").copy()

    team_history = {}  # team -> list of game dicts

    feature_rows = []
    for idx, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        game_date = row["game_date"]

        home_stats = _get_team_rolling(team_history.get(home, []), window)
        away_stats = _get_team_rolling(team_history.get(away, []), window)

        if not home_stats or not away_stats:
            _record_game(team_history, row)
            continue

        features = _build_feature_vector(home_stats, away_stats, home)

        # Metadata + targets
        features["game_id"] = row.get("game_id")
        features["game_date"] = game_date
        features["home_team"] = home
        features["away_team"] = away
        features["season"] = row.get("season", pd.Timestamp(game_date).year)

        # 1st inning results (targets)
        i1_home = row.get("i1_home_runs", 0)
        i1_away = row.get("i1_away_runs", 0)
        features["i1_home_runs"] = i1_home
        features["i1_away_runs"] = i1_away
        features["i1_total"] = i1_home + i1_away

        feature_rows.append(features)

        # Record game AFTER (no leakage)
        _record_game(team_history, row)

    result = pd.DataFrame(feature_rows)
    if not result.empty:
        result["game_date"] = pd.to_datetime(result["game_date"])
    log.info(f"Built {len(result)} first-inning feature vectors")
    return result


def _get_team_rolling(games: list, window: int) -> dict:
    """Compute rolling stats from a team's recent games, including 1st inning."""
    if len(games) < 10:
        return {}

    recent = games[-window:]

    rpg = np.mean([g["rs"] for g in recent])
    rapg = np.mean([g["ra"] for g in recent])
    win_pct = np.mean([g["won"] for g in recent])

    # 1st inning specific stats
    i1_scored = [g["i1_rs"] for g in recent]
    i1_allowed = [g["i1_ra"] for g in recent]
    i1_rpg = np.mean(i1_scored)
    i1_rapg = np.mean(i1_allowed)
    i1_scored_pct = np.mean([1 if r > 0 else 0 for r in i1_scored])
    i1_allowed_pct = np.mean([1 if r > 0 else 0 for r in i1_allowed])

    # Approximate pitching/batting from runs
    era = rapg
    whip = 1.30 + (rapg - 4.5) * 0.05
    ops = 0.720 + (rpg - 4.5) * 0.03

    return {
        "rpg": rpg,
        "rapg": rapg,
        "win_pct": win_pct,
        "ops": ops,
        "era": era,
        "whip": whip,
        "i1_rpg": i1_rpg,
        "i1_rapg": i1_rapg,
        "i1_scored_pct": i1_scored_pct,
        "i1_allowed_pct": i1_allowed_pct,
    }


def _build_feature_vector(home_stats: dict, away_stats: dict, home_team: str) -> dict:
    """Build the feature vector for a 1st inning prediction."""
    pf = PARK_FACTORS.get(home_team, 100) / 100.0

    return {
        # SP placeholders (filled from live data when available)
        "home_sp_era": home_stats["era"],
        "away_sp_era": away_stats["era"],
        "home_sp_whip": home_stats["whip"],
        "away_sp_whip": away_stats["whip"],
        "home_sp_k_per_9": 8.5,
        "away_sp_k_per_9": 8.5,
        "home_sp_bb_per_9": 3.2,
        "away_sp_bb_per_9": 3.2,
        "home_sp_hr_per_9": 1.2,
        "away_sp_hr_per_9": 1.2,
        # 1st inning specific
        "home_1st_inn_rpg": home_stats["i1_rpg"],
        "away_1st_inn_rpg": away_stats["i1_rpg"],
        "home_1st_inn_rapg": home_stats["i1_rapg"],
        "away_1st_inn_rapg": away_stats["i1_rapg"],
        "home_1st_inn_scored_pct": home_stats["i1_scored_pct"],
        "away_1st_inn_scored_pct": away_stats["i1_scored_pct"],
        "home_1st_inn_allowed_pct": home_stats["i1_allowed_pct"],
        "away_1st_inn_allowed_pct": away_stats["i1_allowed_pct"],
        # Team offensive
        "home_rpg": home_stats["rpg"],
        "away_rpg": away_stats["rpg"],
        "home_ops": home_stats["ops"],
        "away_ops": away_stats["ops"],
        # Combined
        "combined_1st_inn_rpg": home_stats["i1_rpg"] + away_stats["i1_rpg"],
        "combined_1st_inn_score_pct": (home_stats["i1_scored_pct"] + away_stats["i1_scored_pct"]) / 2,
        "park_factor": pf,
        "sp_era_combined": (home_stats["era"] + away_stats["era"]) / 2,
        "sp_whip_combined": (home_stats["whip"] + away_stats["whip"]) / 2,
        # Momentum
        "home_win_pct": home_stats["win_pct"],
        "away_win_pct": away_stats["win_pct"],
    }


def _record_game(team_history: dict, row):
    """Record a game in both teams' history."""
    home = row["home_team"]
    away = row["away_team"]
    hs = row["home_score"]
    as_ = row["away_score"]
    i1_h = row.get("i1_home_runs", 0)
    i1_a = row.get("i1_away_runs", 0)

    for team, rs, ra, i1_rs, i1_ra in [
        (home, hs, as_, i1_h, i1_a),
        (away, as_, hs, i1_a, i1_h),
    ]:
        if team not in team_history:
            team_history[team] = []
        team_history[team].append({
            "rs": rs, "ra": ra,
            "won": rs > ra,
            "i1_rs": i1_rs,
            "i1_ra": i1_ra,
        })
