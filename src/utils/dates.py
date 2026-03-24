"""
Date utilities for MLB season handling.
"""

from datetime import date, datetime, timedelta

# Approximate MLB season boundaries (regular season)
SEASON_DATES = {
    2019: (date(2019, 3, 28), date(2019, 9, 29)),
    2020: (date(2020, 7, 23), date(2020, 9, 27)),   # COVID shortened
    2021: (date(2021, 4, 1), date(2021, 10, 3)),
    2022: (date(2022, 4, 7), date(2022, 10, 5)),
    2023: (date(2023, 3, 30), date(2023, 10, 1)),
    2024: (date(2024, 3, 28), date(2024, 9, 29)),
    2025: (date(2025, 3, 27), date(2025, 9, 28)),
    2026: (date(2026, 3, 26), date(2026, 9, 27)),
}


def get_season(d: date) -> int:
    """Return the MLB season year for a given date."""
    return d.year


def season_start(year: int) -> date:
    """Return opening day for a season."""
    if year in SEASON_DATES:
        return SEASON_DATES[year][0]
    # Default estimate: late March
    return date(year, 3, 28)


def season_end(year: int) -> date:
    """Return last day of regular season."""
    if year in SEASON_DATES:
        return SEASON_DATES[year][1]
    return date(year, 9, 29)


def days_into_season(d: date) -> int:
    """Return how many days into the current season a date is."""
    start = season_start(d.year)
    return max(0, (d - start).days)


def is_early_season(d: date, threshold_days: int = 45) -> bool:
    """Check if we're in the early season (stats unreliable)."""
    return days_into_season(d) < threshold_days


def date_range(start: date, end: date) -> list[date]:
    """Generate list of dates from start to end inclusive."""
    days = (end - start).days + 1
    return [start + timedelta(days=i) for i in range(days)]


def format_date(d: date, fmt: str = "%Y-%m-%d") -> str:
    """Format date for API calls."""
    return d.strftime(fmt)


def parse_date(s: str, fmt: str = "%Y-%m-%d") -> date:
    """Parse date string."""
    return datetime.strptime(s, fmt).date()
