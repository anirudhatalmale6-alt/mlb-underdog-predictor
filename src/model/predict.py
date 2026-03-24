"""
Prediction module — generates daily picks from the trained model.
"""

import pandas as pd
import numpy as np
from datetime import date

from config.settings import EDGE_THRESHOLD, MIN_UNDERDOG_ODDS, MAX_UNDERDOG_ODDS
from src.features.builder import FEATURE_COLUMNS
from src.model.registry import load_latest_model
from src.utils.odds_math import american_to_implied, odds_to_decimal, calculate_kelly
from src.utils.logging import get_logger

log = get_logger(__name__)


def generate_predictions(
    games_with_features: list[dict],
    model=None,
    metadata: dict = None,
) -> pd.DataFrame:
    """
    Generate predictions for today's qualifying underdog games.

    Args:
        games_with_features: List of dicts, each containing game info + feature values
        model: Trained model (loads latest if None)
        metadata: Model metadata (loads latest if None)

    Returns:
        DataFrame of picks with win probability, edge, and recommendations
    """
    if model is None:
        model, metadata = load_latest_model()

    if not games_with_features:
        log.info("No qualifying games to predict.")
        return pd.DataFrame()

    # Build feature matrix
    feature_cols = metadata.get("features", FEATURE_COLUMNS)
    available_cols = [c for c in feature_cols if c in games_with_features[0]]

    df = pd.DataFrame(games_with_features)
    X = df[available_cols].fillna(0)

    # Predict
    probs = model.predict_proba(X)[:, 1]

    # Build output
    results = []
    for i, game in enumerate(games_with_features):
        model_prob = float(probs[i])
        underdog_odds = game.get("underdog_odds", 150)
        market_prob = american_to_implied(underdog_odds)
        edge = model_prob - market_prob
        decimal_odds = odds_to_decimal(underdog_odds)
        kelly = calculate_kelly(model_prob, decimal_odds)

        pick = {
            "bet_type": "MONEYLINE",
            "game_date": game.get("game_date", str(date.today())),
            "home_team": game.get("home_team", ""),
            "away_team": game.get("away_team", ""),
            "underdog_team": game.get("underdog_team", ""),
            "home_sp_name": game.get("home_sp_name", "TBD"),
            "away_sp_name": game.get("away_sp_name", "TBD"),
            "underdog_odds": underdog_odds,
            "market_implied_prob": round(market_prob, 4),
            "model_win_prob": round(model_prob, 4),
            "edge": round(edge, 4),
            "edge_pct": f"{edge * 100:.1f}%",
            "kelly_fraction": round(kelly, 4),
            "confidence": _confidence_label(edge),
            "recommended": edge >= EDGE_THRESHOLD,
            "notes": _generate_notes(game, model_prob, edge),
        }
        results.append(pick)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("edge", ascending=False)

    recommended = results_df[results_df["recommended"]].shape[0]
    log.info(f"Generated {len(results_df)} predictions, {recommended} recommended picks")

    return results_df


def _confidence_label(edge: float) -> str:
    """Convert edge to a human-readable confidence label."""
    if edge >= 0.10:
        return "HIGH"
    elif edge >= 0.05:
        return "MEDIUM"
    elif edge >= EDGE_THRESHOLD:
        return "LOW"
    else:
        return "NO PLAY"


def _generate_notes(game: dict, model_prob: float, edge: float) -> str:
    """Generate brief analytical notes for a pick."""
    notes = []

    # SP matchup notes
    ud_era = game.get("ud_sp_era", 4.5)
    fav_era = game.get("fav_sp_era", 4.5)
    if ud_era < fav_era:
        notes.append("Underdog SP has better ERA")
    elif fav_era < 3.5:
        notes.append("Facing elite SP")

    # Momentum notes
    ud_streak = game.get("ud_mom_streak", 0)
    if ud_streak >= 3:
        notes.append(f"Underdog on {ud_streak}W streak")
    elif ud_streak <= -4:
        notes.append(f"Underdog on {abs(ud_streak)}L skid")

    fav_streak = game.get("fav_mom_streak", 0)
    if fav_streak <= -3:
        notes.append(f"Favorite on {abs(fav_streak)}L skid")

    # Value note
    if edge >= 0.08:
        notes.append("Strong value play")
    elif edge >= 0.05:
        notes.append("Good value")

    # Home underdog
    if game.get("underdog_is_home", 0) == 1:
        notes.append("Home underdog")

    return "; ".join(notes) if notes else "Standard play"
