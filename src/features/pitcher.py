"""
Starting pitcher feature extraction.
Computes pitcher quality metrics from season stats and recent game logs.
"""

import numpy as np
from config.settings import FIP_CONSTANT, SP_RECENT_STARTS


def compute_pitcher_features(season_stats: dict, game_logs: list[dict] = None) -> dict:
    """
    Compute features for a starting pitcher.

    Args:
        season_stats: Dict from mlb_statsapi.get_pitcher_season_stats()
        game_logs: List from mlb_statsapi.get_pitcher_game_log() (optional, for recent form)

    Returns:
        Dict of pitcher features (all prefixed with sp_)
    """
    if not season_stats:
        return _default_pitcher_features()

    ip = max(season_stats.get("ip", 0), 0.1)  # Avoid division by zero
    k = season_stats.get("strikeouts", 0)
    bb = season_stats.get("walks", 0)
    hr = season_stats.get("home_runs", 0)
    hits = season_stats.get("hits", 0)
    er = season_stats.get("earned_runs", 0)
    gs = max(season_stats.get("games_started", 1), 1)

    features = {
        "sp_era": season_stats.get("era", 4.50),
        "sp_whip": season_stats.get("whip", 1.30),
        "sp_k_per_9": (k / ip) * 9 if ip > 0 else 0,
        "sp_bb_per_9": (bb / ip) * 9 if ip > 0 else 0,
        "sp_hr_per_9": (hr / ip) * 9 if ip > 0 else 0,
        "sp_k_bb_ratio": k / max(bb, 1),
        "sp_fip": _compute_fip(hr, bb, k, ip),
        "sp_ip_per_start": ip / gs,
        "sp_total_ip": ip,
        "sp_games_started": gs,
    }

    # Recent form from game logs
    if game_logs and len(game_logs) > 0:
        recent = game_logs[-SP_RECENT_STARTS:] if len(game_logs) >= SP_RECENT_STARTS else game_logs
        recent_ip = sum(g.get("ip", 0) for g in recent)
        recent_er = sum(g.get("earned_runs", 0) for g in recent)
        recent_k = sum(g.get("strikeouts", 0) for g in recent)
        recent_bb = sum(g.get("walks", 0) for g in recent)

        features["sp_era_recent"] = (recent_er / max(recent_ip, 0.1)) * 9
        features["sp_k_per_9_recent"] = (recent_k / max(recent_ip, 0.1)) * 9
        features["sp_bb_per_9_recent"] = (recent_bb / max(recent_ip, 0.1)) * 9
        features["sp_ip_per_start_recent"] = recent_ip / max(len(recent), 1)

        # Quality start rate (6+ IP, 3- ER per start)
        qs_count = sum(
            1 for g in recent if g.get("ip", 0) >= 6 and g.get("earned_runs", 0) <= 3
        )
        features["sp_qs_rate_recent"] = qs_count / max(len(recent), 1)
    else:
        features["sp_era_recent"] = features["sp_era"]
        features["sp_k_per_9_recent"] = features["sp_k_per_9"]
        features["sp_bb_per_9_recent"] = features["sp_bb_per_9"]
        features["sp_ip_per_start_recent"] = features["sp_ip_per_start"]
        features["sp_qs_rate_recent"] = 0.0

    return features


def _compute_fip(hr: int, bb: int, k: int, ip: float) -> float:
    """Compute Fielding Independent Pitching."""
    if ip <= 0:
        return 4.50
    return ((13 * hr + 3 * bb - 2 * k) / ip) + FIP_CONSTANT


def _default_pitcher_features() -> dict:
    """Return league-average pitcher features when stats unavailable."""
    return {
        "sp_era": 4.50,
        "sp_whip": 1.30,
        "sp_k_per_9": 8.5,
        "sp_bb_per_9": 3.2,
        "sp_hr_per_9": 1.3,
        "sp_k_bb_ratio": 2.5,
        "sp_fip": 4.20,
        "sp_ip_per_start": 5.2,
        "sp_total_ip": 0,
        "sp_games_started": 0,
        "sp_era_recent": 4.50,
        "sp_k_per_9_recent": 8.5,
        "sp_bb_per_9_recent": 3.2,
        "sp_ip_per_start_recent": 5.2,
        "sp_qs_rate_recent": 0.0,
    }
