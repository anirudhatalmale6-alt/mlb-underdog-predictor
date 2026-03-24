"""
Odds conversion utilities.
Handles American odds <-> implied probability conversions and vig removal.
"""


def american_to_implied(odds: int) -> float:
    """Convert American odds to implied probability (0-1)."""
    if odds > 0:
        return 100.0 / (odds + 100.0)
    else:
        return abs(odds) / (abs(odds) + 100.0)


def implied_to_american(prob: float) -> int:
    """Convert implied probability (0-1) to American odds."""
    if prob <= 0 or prob >= 1:
        raise ValueError(f"Probability must be between 0 and 1, got {prob}")
    if prob >= 0.5:
        return int(round(-prob / (1 - prob) * 100))
    else:
        return int(round((1 - prob) / prob * 100))


def remove_vig(home_odds: int, away_odds: int) -> tuple[float, float]:
    """
    Remove the bookmaker's vig from a two-way moneyline.
    Returns (home_true_prob, away_true_prob) summing to 1.0.
    """
    home_imp = american_to_implied(home_odds)
    away_imp = american_to_implied(away_odds)
    total = home_imp + away_imp  # > 1.0 due to vig
    return home_imp / total, away_imp / total


def is_qualifying_underdog(odds: int, min_odds: int = 130, max_odds: int = 250) -> bool:
    """Check if American odds fall within the qualifying underdog range."""
    return min_odds <= odds <= max_odds


def odds_to_decimal(american_odds: int) -> float:
    """Convert American odds to decimal odds."""
    if american_odds > 0:
        return (american_odds / 100.0) + 1.0
    else:
        return (100.0 / abs(american_odds)) + 1.0


def calculate_edge(model_prob: float, market_odds: int) -> float:
    """
    Calculate the model's edge over the market.
    Positive edge = model thinks underdog wins more often than market implies.
    """
    market_prob = american_to_implied(market_odds)
    return model_prob - market_prob


def calculate_kelly(model_prob: float, decimal_odds: float) -> float:
    """
    Kelly criterion for optimal bet sizing.
    Returns fraction of bankroll to wager (0 if no edge).
    """
    q = 1 - model_prob
    b = decimal_odds - 1  # net odds
    kelly = (model_prob * b - q) / b
    return max(0.0, kelly)


def elo_to_implied(elo_diff: float) -> float:
    """
    Convert Elo rating difference to win probability.
    elo_diff = team_elo - opponent_elo (positive = favored).
    """
    return 1.0 / (1.0 + 10 ** (-elo_diff / 400.0))


def elo_to_american(elo_diff: float) -> int:
    """Convert Elo difference to American odds."""
    prob = elo_to_implied(elo_diff)
    return implied_to_american(prob)
