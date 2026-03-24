"""
Feature builder for MLB totals (over/under) models.
Builds features for both full-game and first-5-innings totals.
"""

import numpy as np
import pandas as pd

from config.settings import PARK_FACTORS
from src.utils.logging import get_logger

log = get_logger(__name__)

# Features used by both totals models
TOTALS_FEATURE_COLUMNS = [
    # Team offensive rolling stats
    "home_rpg", "away_rpg",
    "home_ops", "away_ops",
    "home_iso", "away_iso",
    "home_bat_k_rate", "away_bat_k_rate",
    "home_bat_bb_rate", "away_bat_bb_rate",
    # Team pitching rolling stats
    "home_rapg", "away_rapg",
    "home_era", "away_era",
    "home_whip", "away_whip",
    # SP stats (when available)
    "home_sp_era", "away_sp_era",
    "home_sp_whip", "away_sp_whip",
    "home_sp_k_per_9", "away_sp_k_per_9",
    "home_sp_bb_per_9", "away_sp_bb_per_9",
    "home_sp_ip_per_start", "away_sp_ip_per_start",
    # Combined/derived
    "combined_rpg",         # home_rpg + away_rpg
    "combined_rapg",        # home_rapg + away_rapg
    "combined_ops",         # home_ops + away_ops
    "park_factor",
    "sp_era_combined",      # average of both SPs
    "sp_whip_combined",
    # Rolling game totals
    "home_recent_total_avg",  # avg total runs in recent home team games
    "away_recent_total_avg",
    # Bullpen
    "home_bp_era", "away_bp_era",
    # Momentum/form
    "home_win_pct", "away_win_pct",
    "home_run_diff_pg", "away_run_diff_pg",
]

# F5-specific features (starter-heavy)
F5_FEATURE_COLUMNS = TOTALS_FEATURE_COLUMNS + [
    "home_f5_rpg", "away_f5_rpg",
    "combined_f5_rpg",
    "home_recent_f5_avg",
    "away_recent_f5_avg",
]


def build_totals_features_historical(games_df: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    """
    Build totals features from historical game data.
    Games must have: home_score, away_score, total_runs, f5_total, home_team, away_team.
    """
    df = games_df.sort_values("game_date").copy()

    # Build per-team rolling stats
    team_history = {}  # team -> list of game dicts

    feature_rows = []
    for idx, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        game_date = row["game_date"]

        # Get pre-game rolling stats
        home_stats = _get_team_rolling(team_history.get(home, []), window)
        away_stats = _get_team_rolling(team_history.get(away, []), window)

        if not home_stats or not away_stats:
            # Not enough history yet — record game and skip
            _record_game(team_history, row)
            continue

        features = _build_totals_vector(home_stats, away_stats, home)

        # Metadata + targets
        features["game_id"] = row.get("game_id")
        features["game_date"] = game_date
        features["home_team"] = home
        features["away_team"] = away
        features["season"] = row.get("season", pd.Timestamp(game_date).year)
        features["total_runs"] = row.get("total_runs", row["home_score"] + row["away_score"])
        features["f5_total"] = row.get("f5_total", 0)
        features["home_score"] = row["home_score"]
        features["away_score"] = row["away_score"]

        feature_rows.append(features)

        # Record game AFTER extracting features (no leakage)
        _record_game(team_history, row)

    result = pd.DataFrame(feature_rows)
    if not result.empty:
        result["game_date"] = pd.to_datetime(result["game_date"])
    log.info(f"Built {len(result)} totals feature vectors")
    return result


def _get_team_rolling(games: list, window: int) -> dict:
    """Compute rolling stats from a team's recent games."""
    if len(games) < 10:
        return {}

    recent = games[-window:]
    last10 = games[-10:]

    rpg = np.mean([g["rs"] for g in recent])
    rapg = np.mean([g["ra"] for g in recent])
    total_avg = np.mean([g["rs"] + g["ra"] for g in recent])
    f5_rpg = np.mean([g.get("f5_rs", g["rs"] * 0.55) for g in recent])
    f5_avg = np.mean([g.get("f5_total", (g["rs"] + g["ra"]) * 0.55) for g in recent])
    win_pct = np.mean([g["won"] for g in recent])
    run_diff = rpg - rapg

    # Approximate pitching stats from runs allowed
    era = rapg  # RA/G as ERA proxy
    whip = 1.30 + (rapg - 4.5) * 0.05

    # Approximate batting from runs scored
    ops = 0.720 + (rpg - 4.5) * 0.03
    iso = 0.157 + (rpg - 4.5) * 0.01
    bat_k_rate = 0.225
    bat_bb_rate = 0.085

    bp_era = rapg * 1.05  # Bullpen slightly worse than team

    return {
        "rpg": rpg,
        "rapg": rapg,
        "total_avg": total_avg,
        "f5_rpg": f5_rpg,
        "f5_avg": f5_avg,
        "win_pct": win_pct,
        "run_diff_pg": run_diff,
        "era": era,
        "whip": whip,
        "ops": ops,
        "iso": iso,
        "bat_k_rate": bat_k_rate,
        "bat_bb_rate": bat_bb_rate,
        "bp_era": bp_era,
    }


def _build_totals_vector(home_stats: dict, away_stats: dict, home_team: str) -> dict:
    """Build the feature vector for a totals prediction."""
    pf = PARK_FACTORS.get(home_team, 100) / 100.0

    return {
        # Team offensive
        "home_rpg": home_stats["rpg"],
        "away_rpg": away_stats["rpg"],
        "home_ops": home_stats["ops"],
        "away_ops": away_stats["ops"],
        "home_iso": home_stats["iso"],
        "away_iso": away_stats["iso"],
        "home_bat_k_rate": home_stats["bat_k_rate"],
        "away_bat_k_rate": away_stats["bat_k_rate"],
        "home_bat_bb_rate": home_stats["bat_bb_rate"],
        "away_bat_bb_rate": away_stats["bat_bb_rate"],
        # Team pitching
        "home_rapg": home_stats["rapg"],
        "away_rapg": away_stats["rapg"],
        "home_era": home_stats["era"],
        "away_era": away_stats["era"],
        "home_whip": home_stats["whip"],
        "away_whip": away_stats["whip"],
        # SP placeholders (filled from live data when available)
        "home_sp_era": home_stats["era"],
        "away_sp_era": away_stats["era"],
        "home_sp_whip": home_stats["whip"],
        "away_sp_whip": away_stats["whip"],
        "home_sp_k_per_9": 8.5,
        "away_sp_k_per_9": 8.5,
        "home_sp_bb_per_9": 3.2,
        "away_sp_bb_per_9": 3.2,
        "home_sp_ip_per_start": 5.2,
        "away_sp_ip_per_start": 5.2,
        # Combined
        "combined_rpg": home_stats["rpg"] + away_stats["rpg"],
        "combined_rapg": home_stats["rapg"] + away_stats["rapg"],
        "combined_ops": home_stats["ops"] + away_stats["ops"],
        "park_factor": pf,
        "sp_era_combined": (home_stats["era"] + away_stats["era"]) / 2,
        "sp_whip_combined": (home_stats["whip"] + away_stats["whip"]) / 2,
        # Rolling totals
        "home_recent_total_avg": home_stats["total_avg"],
        "away_recent_total_avg": away_stats["total_avg"],
        # Bullpen
        "home_bp_era": home_stats["bp_era"],
        "away_bp_era": away_stats["bp_era"],
        # Momentum
        "home_win_pct": home_stats["win_pct"],
        "away_win_pct": away_stats["win_pct"],
        "home_run_diff_pg": home_stats["run_diff_pg"],
        "away_run_diff_pg": away_stats["run_diff_pg"],
        # F5
        "home_f5_rpg": home_stats["f5_rpg"],
        "away_f5_rpg": away_stats["f5_rpg"],
        "combined_f5_rpg": home_stats["f5_rpg"] + away_stats["f5_rpg"],
        "home_recent_f5_avg": home_stats["f5_avg"],
        "away_recent_f5_avg": away_stats["f5_avg"],
    }


def _record_game(team_history: dict, row):
    """Record a game in both teams' history."""
    home = row["home_team"]
    away = row["away_team"]
    hs = row["home_score"]
    as_ = row["away_score"]
    f5_h = row.get("f5_home_score", int(hs * 0.55))
    f5_a = row.get("f5_away_score", int(as_ * 0.55))
    f5_total = row.get("f5_total", f5_h + f5_a)

    for team, rs, ra, f5_rs, opp in [
        (home, hs, as_, f5_h, away),
        (away, as_, hs, f5_a, home),
    ]:
        if team not in team_history:
            team_history[team] = []
        team_history[team].append({
            "rs": rs, "ra": ra,
            "won": rs > ra,
            "f5_rs": f5_rs,
            "f5_total": f5_total,
            "opponent": opp,
        })
