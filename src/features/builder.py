"""
Feature builder — the integration hub.
Assembles all feature sources into a single feature vector per game.
Used both for live predictions and historical backtesting.
"""

import numpy as np
import pandas as pd
from typing import Optional

from config.settings import PARK_FACTORS, EARLY_SEASON_BLEND_DAYS
from src.features.pitcher import compute_pitcher_features, _default_pitcher_features
from src.features.team_batting import compute_batting_features, _default_batting_features
from src.features.bullpen import compute_bullpen_features, _default_bullpen_features
from src.features.momentum import compute_momentum_features, _default_momentum_features
from src.utils.logging import get_logger

log = get_logger(__name__)

# All feature columns output by the builder
FEATURE_COLUMNS = [
    # Underdog SP features
    "ud_sp_era", "ud_sp_whip", "ud_sp_k_per_9", "ud_sp_bb_per_9",
    "ud_sp_hr_per_9", "ud_sp_k_bb_ratio", "ud_sp_fip",
    "ud_sp_ip_per_start", "ud_sp_era_recent", "ud_sp_k_per_9_recent",
    "ud_sp_qs_rate_recent",
    # Favorite SP features
    "fav_sp_era", "fav_sp_whip", "fav_sp_k_per_9", "fav_sp_bb_per_9",
    "fav_sp_hr_per_9", "fav_sp_k_bb_ratio", "fav_sp_fip",
    "fav_sp_ip_per_start", "fav_sp_era_recent", "fav_sp_k_per_9_recent",
    "fav_sp_qs_rate_recent",
    # SP differentials
    "delta_sp_era", "delta_sp_fip", "delta_sp_k_per_9", "delta_sp_whip",
    # Underdog batting
    "ud_bat_ops", "ud_bat_iso", "ud_bat_k_rate", "ud_bat_bb_rate",
    "ud_bat_runs_per_game",
    # Favorite batting
    "fav_bat_ops", "fav_bat_iso", "fav_bat_k_rate", "fav_bat_bb_rate",
    "fav_bat_runs_per_game",
    # Batting differentials
    "delta_bat_ops", "delta_bat_runs_per_game",
    # Bullpen
    "ud_bp_era", "ud_bp_k_per_9", "fav_bp_era", "fav_bp_k_per_9",
    "delta_bp_era",
    # Momentum
    "ud_mom_win_pct", "ud_mom_pyth_pct", "ud_mom_streak",
    "ud_mom_last10_pct", "ud_mom_run_diff_per_game",
    "fav_mom_win_pct", "fav_mom_pyth_pct", "fav_mom_streak",
    "fav_mom_last10_pct", "fav_mom_run_diff_per_game",
    "delta_mom_win_pct", "delta_mom_pyth_pct",
    # Contextual
    "park_factor", "underdog_is_home",
    # Market
    "underdog_odds", "implied_prob_market",
]


def build_feature_vector(
    underdog_sp_season: dict,
    favorite_sp_season: dict,
    underdog_batting: dict,
    favorite_batting: dict,
    underdog_bullpen_pitching: dict,
    favorite_bullpen_pitching: dict,
    underdog_sp_raw: dict,
    favorite_sp_raw: dict,
    underdog_standings: dict,
    favorite_standings: dict,
    underdog_sp_logs: list = None,
    favorite_sp_logs: list = None,
    home_team: str = "",
    underdog_side: str = "away",
    underdog_odds: int = 150,
    market_implied_prob: float = 0.4,
) -> dict:
    """
    Build a complete feature vector for one game.
    All features are from the underdog's perspective.
    """
    # Pitcher features
    ud_sp = compute_pitcher_features(underdog_sp_season, underdog_sp_logs)
    fav_sp = compute_pitcher_features(favorite_sp_season, favorite_sp_logs)

    # Batting features
    ud_bat = compute_batting_features(underdog_batting)
    fav_bat = compute_batting_features(favorite_batting)

    # Bullpen features
    ud_bp = compute_bullpen_features(underdog_bullpen_pitching, underdog_sp_raw)
    fav_bp = compute_bullpen_features(favorite_bullpen_pitching, favorite_sp_raw)

    # Momentum features
    ud_mom = compute_momentum_features(underdog_standings)
    fav_mom = compute_momentum_features(favorite_standings)

    # Park factor (use home team's park)
    pf = PARK_FACTORS.get(home_team, 100) / 100.0

    features = {
        # Underdog SP
        "ud_sp_era": ud_sp["sp_era"],
        "ud_sp_whip": ud_sp["sp_whip"],
        "ud_sp_k_per_9": ud_sp["sp_k_per_9"],
        "ud_sp_bb_per_9": ud_sp["sp_bb_per_9"],
        "ud_sp_hr_per_9": ud_sp["sp_hr_per_9"],
        "ud_sp_k_bb_ratio": ud_sp["sp_k_bb_ratio"],
        "ud_sp_fip": ud_sp["sp_fip"],
        "ud_sp_ip_per_start": ud_sp["sp_ip_per_start"],
        "ud_sp_era_recent": ud_sp["sp_era_recent"],
        "ud_sp_k_per_9_recent": ud_sp["sp_k_per_9_recent"],
        "ud_sp_qs_rate_recent": ud_sp["sp_qs_rate_recent"],

        # Favorite SP
        "fav_sp_era": fav_sp["sp_era"],
        "fav_sp_whip": fav_sp["sp_whip"],
        "fav_sp_k_per_9": fav_sp["sp_k_per_9"],
        "fav_sp_bb_per_9": fav_sp["sp_bb_per_9"],
        "fav_sp_hr_per_9": fav_sp["sp_hr_per_9"],
        "fav_sp_k_bb_ratio": fav_sp["sp_k_bb_ratio"],
        "fav_sp_fip": fav_sp["sp_fip"],
        "fav_sp_ip_per_start": fav_sp["sp_ip_per_start"],
        "fav_sp_era_recent": fav_sp["sp_era_recent"],
        "fav_sp_k_per_9_recent": fav_sp["sp_k_per_9_recent"],
        "fav_sp_qs_rate_recent": fav_sp["sp_qs_rate_recent"],

        # SP differentials (positive = underdog SP is worse)
        "delta_sp_era": ud_sp["sp_era"] - fav_sp["sp_era"],
        "delta_sp_fip": ud_sp["sp_fip"] - fav_sp["sp_fip"],
        "delta_sp_k_per_9": ud_sp["sp_k_per_9"] - fav_sp["sp_k_per_9"],
        "delta_sp_whip": ud_sp["sp_whip"] - fav_sp["sp_whip"],

        # Underdog batting
        "ud_bat_ops": ud_bat["bat_ops"],
        "ud_bat_iso": ud_bat["bat_iso"],
        "ud_bat_k_rate": ud_bat["bat_k_rate"],
        "ud_bat_bb_rate": ud_bat["bat_bb_rate"],
        "ud_bat_runs_per_game": ud_bat["bat_runs_per_game"],

        # Favorite batting
        "fav_bat_ops": fav_bat["bat_ops"],
        "fav_bat_iso": fav_bat["bat_iso"],
        "fav_bat_k_rate": fav_bat["bat_k_rate"],
        "fav_bat_bb_rate": fav_bat["bat_bb_rate"],
        "fav_bat_runs_per_game": fav_bat["bat_runs_per_game"],

        # Batting differentials
        "delta_bat_ops": ud_bat["bat_ops"] - fav_bat["bat_ops"],
        "delta_bat_runs_per_game": ud_bat["bat_runs_per_game"] - fav_bat["bat_runs_per_game"],

        # Bullpen
        "ud_bp_era": ud_bp["bp_era"],
        "ud_bp_k_per_9": ud_bp["bp_k_per_9"],
        "fav_bp_era": fav_bp["bp_era"],
        "fav_bp_k_per_9": fav_bp["bp_k_per_9"],
        "delta_bp_era": ud_bp["bp_era"] - fav_bp["bp_era"],

        # Momentum
        "ud_mom_win_pct": ud_mom["mom_win_pct"],
        "ud_mom_pyth_pct": ud_mom["mom_pyth_pct"],
        "ud_mom_streak": ud_mom["mom_streak"],
        "ud_mom_last10_pct": ud_mom["mom_last10_pct"],
        "ud_mom_run_diff_per_game": ud_mom["mom_run_diff_per_game"],
        "fav_mom_win_pct": fav_mom["mom_win_pct"],
        "fav_mom_pyth_pct": fav_mom["mom_pyth_pct"],
        "fav_mom_streak": fav_mom["mom_streak"],
        "fav_mom_last10_pct": fav_mom["mom_last10_pct"],
        "fav_mom_run_diff_per_game": fav_mom["mom_run_diff_per_game"],
        "delta_mom_win_pct": ud_mom["mom_win_pct"] - fav_mom["mom_win_pct"],
        "delta_mom_pyth_pct": ud_mom["mom_pyth_pct"] - fav_mom["mom_pyth_pct"],

        # Contextual
        "park_factor": pf,
        "underdog_is_home": 1 if underdog_side == "home" else 0,

        # Market
        "underdog_odds": underdog_odds,
        "implied_prob_market": market_implied_prob,
    }

    return features


def build_historical_features(games_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build feature matrix from historical games DataFrame.
    Uses Elo-derived metrics as proxies since we can't query the API for
    historical per-game pitcher stats (too many API calls).

    For backtesting, we use simplified features derived from the
    cumulative Elo ratings and game results.
    """
    from src.utils.odds_math import american_to_implied

    log.info(f"Building features for {len(games_df)} historical games...")

    # We need rolling stats per team computed from game results
    # Sort by date
    df = games_df.sort_values("game_date").copy()

    # Compute rolling stats for each team
    team_stats = _compute_rolling_team_stats(df)

    feature_rows = []
    for idx, row in df.iterrows():
        game_date = row["game_date"]
        home = row["home_team"]
        away = row["away_team"]
        underdog_side = row.get("underdog", "away")
        ud_team = home if underdog_side == "home" else away
        fav_team = away if underdog_side == "home" else home
        ud_odds = row.get("underdog_odds", 150)

        # Get rolling stats for both teams (as of before this game)
        ud_stats = team_stats.get((ud_team, str(game_date)[:10]), {})
        fav_stats = team_stats.get((fav_team, str(game_date)[:10]), {})

        if not ud_stats or not fav_stats:
            continue

        # Market implied probability
        try:
            mkt_prob = american_to_implied(int(ud_odds))
        except (ValueError, ZeroDivisionError):
            mkt_prob = 0.4

        pf = PARK_FACTORS.get(home, 100) / 100.0

        features = {
            # SP proxies from team pitching rolling stats
            "ud_sp_era": ud_stats.get("era", 4.50),
            "ud_sp_whip": ud_stats.get("whip", 1.30),
            "ud_sp_k_per_9": ud_stats.get("k_per_9", 8.5),
            "ud_sp_bb_per_9": ud_stats.get("bb_per_9", 3.2),
            "ud_sp_hr_per_9": 1.3,
            "ud_sp_k_bb_ratio": ud_stats.get("k_per_9", 8.5) / max(ud_stats.get("bb_per_9", 3.2), 0.1),
            "ud_sp_fip": ud_stats.get("era", 4.50) * 0.95,  # Proxy
            "ud_sp_ip_per_start": 5.2,
            "ud_sp_era_recent": ud_stats.get("era_recent", 4.50),
            "ud_sp_k_per_9_recent": ud_stats.get("k_per_9", 8.5),
            "ud_sp_qs_rate_recent": 0.3,

            "fav_sp_era": fav_stats.get("era", 4.50),
            "fav_sp_whip": fav_stats.get("whip", 1.30),
            "fav_sp_k_per_9": fav_stats.get("k_per_9", 8.5),
            "fav_sp_bb_per_9": fav_stats.get("bb_per_9", 3.2),
            "fav_sp_hr_per_9": 1.3,
            "fav_sp_k_bb_ratio": fav_stats.get("k_per_9", 8.5) / max(fav_stats.get("bb_per_9", 3.2), 0.1),
            "fav_sp_fip": fav_stats.get("era", 4.50) * 0.95,
            "fav_sp_ip_per_start": 5.2,
            "fav_sp_era_recent": fav_stats.get("era_recent", 4.50),
            "fav_sp_k_per_9_recent": fav_stats.get("k_per_9", 8.5),
            "fav_sp_qs_rate_recent": 0.3,

            "delta_sp_era": ud_stats.get("era", 4.50) - fav_stats.get("era", 4.50),
            "delta_sp_fip": (ud_stats.get("era", 4.50) - fav_stats.get("era", 4.50)) * 0.95,
            "delta_sp_k_per_9": ud_stats.get("k_per_9", 8.5) - fav_stats.get("k_per_9", 8.5),
            "delta_sp_whip": ud_stats.get("whip", 1.30) - fav_stats.get("whip", 1.30),

            "ud_bat_ops": ud_stats.get("ops", 0.720),
            "ud_bat_iso": ud_stats.get("iso", 0.157),
            "ud_bat_k_rate": ud_stats.get("bat_k_rate", 0.225),
            "ud_bat_bb_rate": ud_stats.get("bat_bb_rate", 0.085),
            "ud_bat_runs_per_game": ud_stats.get("rpg", 4.5),

            "fav_bat_ops": fav_stats.get("ops", 0.720),
            "fav_bat_iso": fav_stats.get("iso", 0.157),
            "fav_bat_k_rate": fav_stats.get("bat_k_rate", 0.225),
            "fav_bat_bb_rate": fav_stats.get("bat_bb_rate", 0.085),
            "fav_bat_runs_per_game": fav_stats.get("rpg", 4.5),

            "delta_bat_ops": ud_stats.get("ops", 0.720) - fav_stats.get("ops", 0.720),
            "delta_bat_runs_per_game": ud_stats.get("rpg", 4.5) - fav_stats.get("rpg", 4.5),

            "ud_bp_era": ud_stats.get("bp_era", 4.20),
            "ud_bp_k_per_9": ud_stats.get("k_per_9", 9.0),
            "fav_bp_era": fav_stats.get("bp_era", 4.20),
            "fav_bp_k_per_9": fav_stats.get("k_per_9", 9.0),
            "delta_bp_era": ud_stats.get("bp_era", 4.20) - fav_stats.get("bp_era", 4.20),

            "ud_mom_win_pct": ud_stats.get("win_pct", 0.500),
            "ud_mom_pyth_pct": ud_stats.get("pyth_pct", 0.500),
            "ud_mom_streak": ud_stats.get("streak", 0),
            "ud_mom_last10_pct": ud_stats.get("last10_pct", 0.500),
            "ud_mom_run_diff_per_game": ud_stats.get("run_diff_pg", 0.0),
            "fav_mom_win_pct": fav_stats.get("win_pct", 0.500),
            "fav_mom_pyth_pct": fav_stats.get("pyth_pct", 0.500),
            "fav_mom_streak": fav_stats.get("streak", 0),
            "fav_mom_last10_pct": fav_stats.get("last10_pct", 0.500),
            "fav_mom_run_diff_per_game": fav_stats.get("run_diff_pg", 0.0),
            "delta_mom_win_pct": ud_stats.get("win_pct", 0.500) - fav_stats.get("win_pct", 0.500),
            "delta_mom_pyth_pct": ud_stats.get("pyth_pct", 0.500) - fav_stats.get("pyth_pct", 0.500),

            "park_factor": pf,
            "underdog_is_home": 1 if underdog_side == "home" else 0,
            "underdog_odds": ud_odds,
            "implied_prob_market": mkt_prob,
        }

        feature_rows.append({
            "game_id": row.get("game_id"),
            "game_date": game_date,
            "home_team": home,
            "away_team": away,
            "underdog_team": ud_team,
            "underdog_won": row.get("underdog_won", 0),
            **features,
        })

    result = pd.DataFrame(feature_rows)
    log.info(f"Built {len(result)} feature vectors from historical games")
    return result


def _compute_rolling_team_stats(df: pd.DataFrame, window: int = 30) -> dict:
    """
    Compute rolling team stats from game results.
    Returns dict keyed by (team, date_str) -> stats dict.

    This avoids look-ahead bias: stats for game on date D use only
    games strictly before D.
    """
    stats_lookup = {}
    team_games = {}  # team -> list of game dicts (chronological)

    for _, row in df.iterrows():
        date_str = str(row["game_date"])[:10]
        home = row["home_team"]
        away = row["away_team"]
        hs = row["home_score"]
        as_ = row["away_score"]

        # Before processing this game, snapshot current stats
        for team in [home, away]:
            if team in team_games and len(team_games[team]) >= 10:
                games = team_games[team]
                recent = games[-window:]
                last10 = games[-10:]

                wins = sum(1 for g in recent if g["won"])
                total = len(recent)
                rs = sum(g["rs"] for g in recent)
                ra = sum(g["ra"] for g in recent)
                rs_total = max(rs, 1)
                ra_total = max(ra, 1)

                pyth = (rs_total ** 1.83) / (rs_total ** 1.83 + ra_total ** 1.83)

                # Streak
                streak = 0
                for g in reversed(games):
                    if g["won"] and (streak >= 0):
                        streak += 1
                    elif not g["won"] and (streak <= 0):
                        streak -= 1
                    else:
                        break

                l10_wins = sum(1 for g in last10 if g["won"])

                stats_lookup[(team, date_str)] = {
                    "win_pct": wins / max(total, 1),
                    "pyth_pct": pyth,
                    "rpg": rs / max(total, 1),
                    "rapg": ra / max(total, 1),
                    "run_diff_pg": (rs - ra) / max(total, 1),
                    "streak": streak,
                    "last10_pct": l10_wins / 10,
                    "ops": 0.720 + (rs / max(total, 1) - 4.5) * 0.03,  # Proxy
                    "iso": 0.157 + (rs / max(total, 1) - 4.5) * 0.01,
                    "bat_k_rate": 0.225,
                    "bat_bb_rate": 0.085,
                    "era": (ra / max(total, 1)) * 9 / 9,  # RA/G as ERA proxy
                    "whip": 1.30 + (ra / max(total, 1) - 4.5) * 0.05,
                    "k_per_9": 8.5,
                    "bb_per_9": 3.2,
                    "bp_era": (ra / max(total, 1)) * 1.05,  # Slightly worse than team
                    "era_recent": sum(g["ra"] for g in last10) / 10,
                }

        # Record this game for both teams
        for team, opp, rs, ra, is_home in [
            (home, away, hs, as_, True),
            (away, home, as_, hs, False),
        ]:
            if team not in team_games:
                team_games[team] = []
            team_games[team].append({
                "date": date_str,
                "opponent": opp,
                "rs": rs, "ra": ra,
                "won": rs > ra,
                "is_home": is_home,
            })

    return stats_lookup
