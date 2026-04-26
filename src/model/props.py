"""
Player props prediction module.
Generates daily pitcher strikeout and batter hits picks.
"""

import pandas as pd
import numpy as np
from datetime import date

from config.settings import MODELS_DIR
from src.features.props import (
    PITCHER_K_FEATURES, BATTER_HITS_FEATURES, PITCHER_OUTS_FEATURES,
    build_pitcher_k_features, build_batter_hits_features, build_pitcher_outs_features,
)
from src.model.registry import load_latest_model
from src.utils.odds_math import american_to_implied
from src.utils.logging import get_logger

log = get_logger(__name__)

MAX_PROPS_PER_TYPE = 3  # Max recommended picks per prop type
MIN_EDGE = 0.08  # 8% minimum edge for recommendation


def predict_pitcher_k_props(prop_games: list[dict]) -> pd.DataFrame:
    """
    Generate pitcher strikeout over/under predictions.
    prop_games: list of dicts with pitcher features + prop line info.
    """
    try:
        model, metadata = load_latest_model("xgb_pitcher_k")
    except Exception as e:
        log.warning(f"Could not load pitcher K model: {e}")
        return pd.DataFrame()

    if not prop_games:
        return pd.DataFrame()

    feature_cols = metadata.get("features", PITCHER_K_FEATURES)
    available = [c for c in feature_cols if c in prop_games[0]]

    if not available:
        return pd.DataFrame()

    df = pd.DataFrame(prop_games)
    X = df[available].fillna(0)
    probs = model.predict_proba(X)[:, 1]  # P(over)

    results = []
    for i, game in enumerate(prop_games):
        over_prob = float(probs[i])
        is_over = over_prob > 0.5
        pick_prob = over_prob if is_over else (1 - over_prob)
        edge = pick_prob - 0.5

        line = game.get("k_line", 0)
        over_odds = game.get("k_over_odds", -110)
        under_odds = game.get("k_under_odds", -110)
        pick_odds = over_odds if is_over else under_odds

        # Only consider plus-money or close to even
        results.append({
            "bet_type": "PITCHER K",
            "game_date": game.get("game_date", ""),
            "home_team": game.get("home_team", ""),
            "away_team": game.get("away_team", ""),
            "player_name": game.get("pitcher_name", ""),
            "pick": f"OVER {line} Ks" if is_over else f"UNDER {line} Ks",
            "line": str(line),
            "odds": pick_odds,
            "model_prob": round(pick_prob, 4),
            "edge": round(edge, 4),
            "edge_pct": f"{edge * 100:.1f}%",
            "confidence": _confidence_label(edge),
            "recommended": edge >= MIN_EDGE,
            "notes": _k_notes(game, over_prob, edge, is_over),
        })

    results_df = pd.DataFrame(results).sort_values("edge", ascending=False)
    # Cap recommended
    if not results_df.empty:
        rec_mask = results_df["recommended"]
        rec_indices = results_df[rec_mask].head(MAX_PROPS_PER_TYPE).index
        results_df.loc[rec_mask & ~results_df.index.isin(rec_indices), "recommended"] = False

    return results_df


def predict_batter_hits_props(prop_games: list[dict]) -> pd.DataFrame:
    """
    Generate batter hits over/under predictions.
    """
    try:
        model, metadata = load_latest_model("xgb_batter_hits")
    except Exception as e:
        log.warning(f"Could not load batter hits model: {e}")
        return pd.DataFrame()

    if not prop_games:
        return pd.DataFrame()

    feature_cols = metadata.get("features", BATTER_HITS_FEATURES)
    available = [c for c in feature_cols if c in prop_games[0]]

    if not available:
        return pd.DataFrame()

    df = pd.DataFrame(prop_games)
    X = df[available].fillna(0)
    probs = model.predict_proba(X)[:, 1]  # P(over)

    results = []
    for i, game in enumerate(prop_games):
        over_prob = float(probs[i])
        is_over = over_prob > 0.5
        pick_prob = over_prob if is_over else (1 - over_prob)
        edge = pick_prob - 0.5

        line = game.get("hits_line", 0)
        over_odds = game.get("hits_over_odds", -110)
        under_odds = game.get("hits_under_odds", -110)
        pick_odds = over_odds if is_over else under_odds

        results.append({
            "bet_type": "BATTER HITS",
            "game_date": game.get("game_date", ""),
            "home_team": game.get("home_team", ""),
            "away_team": game.get("away_team", ""),
            "player_name": game.get("batter_name", ""),
            "pick": f"OVER {line} hits" if is_over else f"UNDER {line} hits",
            "line": str(line),
            "odds": pick_odds,
            "model_prob": round(pick_prob, 4),
            "edge": round(edge, 4),
            "edge_pct": f"{edge * 100:.1f}%",
            "confidence": _confidence_label(edge),
            "recommended": edge >= MIN_EDGE,
            "notes": _hits_notes(game, over_prob, edge, is_over),
        })

    results_df = pd.DataFrame(results).sort_values("edge", ascending=False)
    # Cap recommended
    if not results_df.empty:
        rec_mask = results_df["recommended"]
        rec_indices = results_df[rec_mask].head(MAX_PROPS_PER_TYPE).index
        results_df.loc[rec_mask & ~results_df.index.isin(rec_indices), "recommended"] = False

    return results_df


def predict_pitcher_outs_props(prop_games: list[dict]) -> pd.DataFrame:
    """
    Generate pitcher outs recorded over/under predictions.
    """
    try:
        model, metadata = load_latest_model("xgb_pitcher_outs")
    except Exception as e:
        log.warning(f"Could not load pitcher outs model: {e}")
        return pd.DataFrame()

    if not prop_games:
        return pd.DataFrame()

    feature_cols = metadata.get("features", PITCHER_OUTS_FEATURES)
    available = [c for c in feature_cols if c in prop_games[0]]

    if not available:
        return pd.DataFrame()

    df = pd.DataFrame(prop_games)
    X = df[available].fillna(0)
    probs = model.predict_proba(X)[:, 1]  # P(over)

    results = []
    for i, game in enumerate(prop_games):
        over_prob = float(probs[i])
        is_over = over_prob > 0.5
        pick_prob = over_prob if is_over else (1 - over_prob)
        edge = pick_prob - 0.5

        line = game.get("outs_line", 0)
        over_odds = game.get("outs_over_odds", -110)
        under_odds = game.get("outs_under_odds", -110)
        pick_odds = over_odds if is_over else under_odds

        results.append({
            "bet_type": "PITCHER OUTS",
            "game_date": game.get("game_date", ""),
            "home_team": game.get("home_team", ""),
            "away_team": game.get("away_team", ""),
            "player_name": game.get("pitcher_name", ""),
            "pick": f"OVER {line} outs" if is_over else f"UNDER {line} outs",
            "line": str(line),
            "odds": pick_odds,
            "model_prob": round(pick_prob, 4),
            "edge": round(edge, 4),
            "edge_pct": f"{edge * 100:.1f}%",
            "confidence": _confidence_label(edge),
            "recommended": edge >= MIN_EDGE,
            "notes": _outs_notes(game, over_prob, edge, is_over),
        })

    results_df = pd.DataFrame(results).sort_values("edge", ascending=False)
    if not results_df.empty:
        rec_mask = results_df["recommended"]
        rec_indices = results_df[rec_mask].head(MAX_PROPS_PER_TYPE).index
        results_df.loc[rec_mask & ~results_df.index.isin(rec_indices), "recommended"] = False

    return results_df


def _confidence_label(edge: float) -> str:
    if edge >= 0.12:
        return "HIGH"
    elif edge >= 0.10:
        return "MEDIUM"
    elif edge >= MIN_EDGE:
        return "LOW"
    else:
        return "NO PLAY"


def _k_notes(game: dict, over_prob: float, edge: float, is_over: bool) -> str:
    notes = []
    k_avg = game.get("k_per_start_avg", 0)
    opp_k_rate = game.get("opp_team_k_rate", 0.22)

    if is_over:
        if opp_k_rate > 0.24:
            notes.append("Facing high-K team")
        if k_avg > 7:
            notes.append("High K average")
    else:
        if opp_k_rate < 0.20:
            notes.append("Facing low-K team")
        if k_avg < 5:
            notes.append("Low K average")

    if edge >= 0.10:
        notes.append("Strong value")

    return "; ".join(notes) if notes else "Standard play"


def _hits_notes(game: dict, over_prob: float, edge: float, is_over: bool) -> str:
    notes = []
    ba = game.get("batting_avg", 0.250)
    trend = game.get("recent_hits_trend", 0)

    if is_over:
        if trend > 0.3:
            notes.append("Hot streak")
        if ba > 0.280:
            notes.append("High BA")
    else:
        if trend < -0.3:
            notes.append("Cold streak")
        if ba < 0.230:
            notes.append("Low BA")

    if edge >= 0.10:
        notes.append("Strong value")

    return "; ".join(notes) if notes else "Standard play"


def _outs_notes(game: dict, over_prob: float, edge: float, is_over: bool) -> str:
    notes = []
    ip_avg = game.get("ip_per_start", 5.0)
    pitches_per_ip = game.get("pitches_per_ip", 16.0)

    if is_over:
        if ip_avg > 6.0:
            notes.append("Deep into games consistently")
        if pitches_per_ip < 15.5:
            notes.append("Efficient pitcher")
    else:
        if ip_avg < 5.0:
            notes.append("Short outings typical")
        if pitches_per_ip > 17.0:
            notes.append("Inefficient pitcher")
        if game.get("bb_rate", 0) > 0.10:
            notes.append("High walk rate")

    if edge >= 0.10:
        notes.append("Strong value")

    return "; ".join(notes) if notes else "Standard play"
