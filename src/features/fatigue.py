"""
Travel / fatigue features.
Detects travel situations, back-to-back games, and schedule density
that may affect team performance.
"""

# Team venue locations (latitude, longitude) for travel distance estimation
# Grouped by rough geographic region for cross-country detection
TEAM_LOCATIONS = {
    # West Coast
    "SEA": (47.59, -122.33), "SF":  (37.78, -122.39), "OAK": (37.75, -122.20),
    "LAD": (34.07, -118.24), "LAA": (33.80, -117.88), "SD":  (32.71, -117.16),
    "ARI": (33.45, -112.07),
    # Mountain
    "COL": (39.76, -104.99),
    # Central
    "MIN": (44.98, -93.28), "MIL": (43.03, -87.97), "CHC": (41.95, -87.66),
    "CWS": (41.83, -87.63), "STL": (38.62, -90.19), "KC":  (39.05, -94.48),
    "TEX": (32.75, -97.08), "HOU": (29.76, -95.36),
    # East
    "DET": (42.34, -83.05), "CLE": (41.50, -81.69), "CIN": (39.10, -84.51),
    "PIT": (40.45, -80.00), "ATL": (33.89, -84.47),
    "TB":  (27.77, -82.65), "MIA": (25.78, -80.22),
    "TOR": (43.64, -79.39), "BOS": (42.35, -71.10),
    "NYY": (40.83, -73.93), "NYM": (40.76, -73.85),
    "BAL": (39.28, -76.62), "WSH": (38.87, -77.01), "PHI": (39.91, -75.17),
}

# Time zones by team (UTC offset during baseball season / EDT)
TEAM_TIMEZONE_OFFSET = {
    "SEA": -7, "SF": -7, "OAK": -7, "LAD": -7, "LAA": -7, "SD": -7, "ARI": -7,
    "COL": -6, "MIN": -5, "MIL": -5, "CHC": -5, "CWS": -5, "STL": -5,
    "KC": -5, "TEX": -5, "HOU": -5, "DET": -4, "CLE": -4, "CIN": -4,
    "PIT": -4, "ATL": -4, "TB": -4, "MIA": -4, "TOR": -4, "BOS": -4,
    "NYY": -4, "NYM": -4, "BAL": -4, "WSH": -4, "PHI": -4,
}


def _haversine_miles(lat1, lon1, lat2, lon2):
    """Approximate distance in miles between two coordinates."""
    from math import radians, sin, cos, sqrt, atan2
    R = 3959  # Earth radius in miles
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return 2 * R * atan2(sqrt(a), sqrt(1 - a))


def _venue_to_team(venue_name: str, team_abbrev: str) -> str:
    """
    Map a venue name to a team abbreviation for location lookup.
    Falls back to the team abbreviation if venue can't be mapped.
    """
    # Known venue -> team mappings (covers most cases)
    VENUE_MAP = {
        "Yankee Stadium": "NYY", "Citi Field": "NYM", "Fenway Park": "BOS",
        "Camden Yards": "BAL", "Oriole Park at Camden Yards": "BAL",
        "Citizens Bank Park": "PHI", "Nationals Park": "WSH",
        "Tropicana Field": "TB", "loanDepot park": "MIA",
        "Truist Park": "ATL", "Great American Ball Park": "CIN",
        "PNC Park": "PIT", "Progressive Field": "CLE",
        "Comerica Park": "DET", "Rogers Centre": "TOR",
        "Wrigley Field": "CHC", "Guaranteed Rate Field": "CWS",
        "American Family Field": "MIL", "Target Field": "MIN",
        "Busch Stadium": "STL", "Kauffman Stadium": "KC",
        "Minute Maid Park": "HOU", "Daikin Park": "HOU", "Globe Life Field": "TEX",
        "Chase Field": "ARI", "Coors Field": "COL",
        "Dodger Stadium": "LAD", "Angel Stadium": "LAA",
        "Oracle Park": "SF", "Oakland Coliseum": "OAK",
        "RingCentral Coliseum": "OAK", "Petco Park": "SD",
        "T-Mobile Park": "SEA",
    }
    return VENUE_MAP.get(venue_name, team_abbrev)


def compute_fatigue_features(recent_games: list[dict], team_abbrev: str,
                              today_venue: str = "", is_home_today: bool = True) -> dict:
    """
    Compute travel/fatigue features for a team based on recent schedule.

    Args:
        recent_games: List of recent game dicts from get_team_recent_schedule()
        team_abbrev: Team abbreviation (e.g., "NYY")
        today_venue: Today's venue name
        is_home_today: Whether the team is at home today

    Returns:
        Dict with fatigue feature values.
    """
    features = _default_fatigue_features()

    if not recent_games:
        return features

    # Games in last 3 days
    games_last_3 = recent_games[-3:] if len(recent_games) >= 3 else recent_games
    features["games_last_3d"] = len(games_last_3)

    # Games in last 5 days
    features["games_last_5d"] = len(recent_games)

    # Consecutive road games
    road_streak = 0
    for g in reversed(recent_games):
        if not g["is_home"]:
            road_streak += 1
        else:
            break
    features["road_game_streak"] = road_streak

    # Had a day off yesterday (rest advantage)
    if recent_games:
        from datetime import datetime, timedelta
        last_game_date = recent_games[-1].get("date", "")
        if last_game_date:
            try:
                last_dt = datetime.strptime(last_game_date, "%Y-%m-%d").date()
                # If we have today's date context, check gap
                # For now, check if there's a gap in the schedule
                if len(recent_games) >= 2:
                    dates = [g["date"] for g in recent_games]
                    unique_dates = sorted(set(dates))
                    if len(unique_dates) >= 2:
                        d1 = datetime.strptime(unique_dates[-1], "%Y-%m-%d").date()
                        d2 = datetime.strptime(unique_dates[-2], "%Y-%m-%d").date()
                        features["had_day_off"] = 1 if (d1 - d2).days > 1 else 0
            except (ValueError, IndexError):
                pass

    # Travel distance: compare last game's venue to today's venue
    if recent_games and today_venue:
        last_game = recent_games[-1]
        last_venue = last_game.get("venue", "")
        # Use home_team from schedule data as fallback for venue mapping
        last_home = last_game.get("home_team", team_abbrev)
        last_loc_team = _venue_to_team(last_venue, last_home)
        today_loc_team = _venue_to_team(today_venue, team_abbrev if is_home_today else "")

        last_coords = TEAM_LOCATIONS.get(last_loc_team)
        today_coords = TEAM_LOCATIONS.get(today_loc_team) or TEAM_LOCATIONS.get(team_abbrev)

        if last_coords and today_coords:
            dist = _haversine_miles(last_coords[0], last_coords[1],
                                    today_coords[0], today_coords[1])
            features["travel_dist_miles"] = dist
            features["is_cross_country"] = 1 if dist > 1500 else 0

    # Timezone change
    if recent_games:
        last_venue_team = _venue_to_team(recent_games[-1].get("venue", ""), team_abbrev)
        today_venue_team = _venue_to_team(today_venue, team_abbrev)
        last_tz = TEAM_TIMEZONE_OFFSET.get(last_venue_team, -5)
        today_tz = TEAM_TIMEZONE_OFFSET.get(today_venue_team, -5)
        features["tz_change_hours"] = abs(today_tz - last_tz)

    # Double-header detection (2 games on same date)
    if recent_games:
        dates = [g["date"] for g in recent_games]
        from collections import Counter
        date_counts = Counter(dates)
        features["had_doubleheader"] = 1 if any(c >= 2 for c in date_counts.values()) else 0

    # Compute fatigue score (composite)
    # Higher = more fatigued
    score = 0.0
    score += features["games_last_3d"] * 0.15  # Dense schedule
    score += features["road_game_streak"] * 0.1  # Road wear
    score += min(features["travel_dist_miles"] / 3000, 1.0) * 0.3  # Travel impact (capped)
    score += features["tz_change_hours"] * 0.15  # Timezone disruption
    score += features["had_doubleheader"] * 0.1  # Recent doubleheader
    score -= features["had_day_off"] * 0.2  # Rest helps
    features["fatigue_score"] = max(0, min(1, score))

    return features


def _default_fatigue_features() -> dict:
    """Default (neutral) fatigue features when no data available."""
    return {
        "games_last_3d": 0,
        "games_last_5d": 0,
        "road_game_streak": 0,
        "had_day_off": 0,
        "travel_dist_miles": 0.0,
        "is_cross_country": 0,
        "tz_change_hours": 0,
        "had_doubleheader": 0,
        "fatigue_score": 0.0,
    }


# Feature columns exported by this module
FATIGUE_FEATURE_COLUMNS = [
    "ud_fatigue_score", "fav_fatigue_score", "delta_fatigue",
    "ud_travel_dist", "fav_travel_dist",
    "ud_road_streak", "fav_road_streak",
    "ud_rest_advantage", "fav_rest_advantage",
]
