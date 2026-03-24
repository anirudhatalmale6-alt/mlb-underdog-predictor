"""
Team batting feature extraction.
"""


def compute_batting_features(batting_stats: dict) -> dict:
    """
    Compute batting features for a team.

    Args:
        batting_stats: Dict from mlb_statsapi.get_team_batting_stats()

    Returns:
        Dict of team batting features (prefixed with bat_)
    """
    if not batting_stats:
        return _default_batting_features()

    games = max(batting_stats.get("games", 1), 1)
    ab = max(batting_stats.get("at_bats", 1), 1)
    runs = batting_stats.get("runs", 0)
    hits = batting_stats.get("hits", 0)
    hr = batting_stats.get("home_runs", 0)
    k = batting_stats.get("strikeouts", 0)
    bb = batting_stats.get("walks", 0)
    sb = batting_stats.get("stolen_bases", 0)

    avg = batting_stats.get("avg", 0.250)
    obp = batting_stats.get("obp", 0.320)
    slg = batting_stats.get("slg", 0.400)
    ops = batting_stats.get("ops", 0.720)

    # Derived metrics
    iso = slg - avg if slg > avg else 0.0  # Isolated power
    pa = ab + bb  # Approximate plate appearances
    k_rate = k / max(pa, 1)
    bb_rate = bb / max(pa, 1)
    runs_per_game = runs / games

    return {
        "bat_avg": avg,
        "bat_obp": obp,
        "bat_slg": slg,
        "bat_ops": ops,
        "bat_iso": iso,
        "bat_k_rate": k_rate,
        "bat_bb_rate": bb_rate,
        "bat_hr_per_game": hr / games,
        "bat_runs_per_game": runs_per_game,
        "bat_sb_per_game": sb / games,
    }


def _default_batting_features() -> dict:
    """League-average batting features."""
    return {
        "bat_avg": 0.248,
        "bat_obp": 0.317,
        "bat_slg": 0.405,
        "bat_ops": 0.722,
        "bat_iso": 0.157,
        "bat_k_rate": 0.225,
        "bat_bb_rate": 0.085,
        "bat_hr_per_game": 1.1,
        "bat_runs_per_game": 4.5,
        "bat_sb_per_game": 0.6,
    }
