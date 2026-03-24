"""
Bullpen feature extraction.
Uses team-level pitching stats minus the starting pitcher contribution
to approximate bullpen quality.
"""


def compute_bullpen_features(
    team_pitching: dict,
    sp_stats: dict = None,
) -> dict:
    """
    Compute bullpen features by subtracting SP contribution from team pitching.

    Args:
        team_pitching: Dict from mlb_statsapi.get_team_pitching_stats()
        sp_stats: Starting pitcher season stats (to subtract from team totals)

    Returns:
        Dict of bullpen features (prefixed with bp_)
    """
    if not team_pitching:
        return _default_bullpen_features()

    team_ip = max(team_pitching.get("ip", 1), 1)
    team_er = team_pitching.get("era", 4.0) * team_ip / 9
    team_k = team_pitching.get("strikeouts", 0)
    team_bb = team_pitching.get("walks", 0)
    team_hits = team_pitching.get("hits", 0)
    team_hr = team_pitching.get("home_runs", 0)

    # Subtract SP contribution if available
    if sp_stats and sp_stats.get("ip", 0) > 0:
        sp_ip = sp_stats.get("ip", 0)
        sp_er = sp_stats.get("earned_runs", 0)
        sp_k = sp_stats.get("strikeouts", 0)
        sp_bb = sp_stats.get("walks", 0)
        sp_hits = sp_stats.get("hits", 0)
        sp_hr = sp_stats.get("home_runs", 0)

        bp_ip = max(team_ip - sp_ip, 1)
        bp_er = max(team_er - sp_er, 0)
        bp_k = max(team_k - sp_k, 0)
        bp_bb = max(team_bb - sp_bb, 0)
        bp_hits = max(team_hits - sp_hits, 0)
        bp_hr = max(team_hr - sp_hr, 0)
    else:
        # Approximate: bullpen is ~40% of team innings
        bp_ip = team_ip * 0.4
        bp_er = team_er * 0.4
        bp_k = team_k * 0.4
        bp_bb = team_bb * 0.4
        bp_hits = team_hits * 0.4
        bp_hr = team_hr * 0.4

    bp_ip = max(bp_ip, 0.1)

    return {
        "bp_era": (bp_er / bp_ip) * 9,
        "bp_whip": (bp_bb + bp_hits) / bp_ip,
        "bp_k_per_9": (bp_k / bp_ip) * 9,
        "bp_bb_per_9": (bp_bb / bp_ip) * 9,
        "bp_hr_per_9": (bp_hr / bp_ip) * 9,
    }


def _default_bullpen_features() -> dict:
    """League-average bullpen features."""
    return {
        "bp_era": 4.20,
        "bp_whip": 1.30,
        "bp_k_per_9": 9.0,
        "bp_bb_per_9": 3.5,
        "bp_hr_per_9": 1.2,
    }
