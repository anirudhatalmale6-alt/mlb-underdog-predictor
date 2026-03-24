"""
Momentum and streak features derived from standings and recent results.
"""

import numpy as np


def compute_momentum_features(standings_record: dict) -> dict:
    """
    Compute momentum features for a team from standings data.

    Args:
        standings_record: Dict from get_standings() for this team

    Returns:
        Dict of momentum features (prefixed with mom_)
    """
    if not standings_record:
        return _default_momentum_features()

    wins = standings_record.get("wins", 0)
    losses = standings_record.get("losses", 0)
    total = max(wins + losses, 1)
    rs = max(standings_record.get("runs_scored", 1), 1)
    ra = max(standings_record.get("runs_allowed", 1), 1)

    # Pythagorean win expectation
    pyth = (rs ** 1.83) / (rs ** 1.83 + ra ** 1.83)

    # Streak
    streak_type = standings_record.get("streak_type", "")
    streak_num = standings_record.get("streak_number", 0)
    streak_val = streak_num if streak_type == "W" else -streak_num

    # Last 10
    last10_w = standings_record.get("last_ten_wins", 5)
    last10_l = standings_record.get("last_ten_losses", 5)

    # Home/away records
    hw = standings_record.get("home_wins", 0)
    hl = standings_record.get("home_losses", 0)
    aw = standings_record.get("away_wins", 0)
    al = standings_record.get("away_losses", 0)

    return {
        "mom_win_pct": wins / total,
        "mom_pyth_pct": pyth,
        "mom_pyth_diff": (wins / total) - pyth,  # Luck factor
        "mom_streak": streak_val,
        "mom_last10_pct": last10_w / max(last10_w + last10_l, 1),
        "mom_run_diff_per_game": (rs - ra) / total,
        "mom_home_pct": hw / max(hw + hl, 1),
        "mom_away_pct": aw / max(aw + al, 1),
    }


def _default_momentum_features() -> dict:
    """Default momentum features (league average)."""
    return {
        "mom_win_pct": 0.500,
        "mom_pyth_pct": 0.500,
        "mom_pyth_diff": 0.0,
        "mom_streak": 0,
        "mom_last10_pct": 0.500,
        "mom_run_diff_per_game": 0.0,
        "mom_home_pct": 0.500,
        "mom_away_pct": 0.500,
    }
