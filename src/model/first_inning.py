"""
1st inning model training and prediction.
Two models: 1st Inning Moneyline and 1st Inning Total (over/under 0.5 runs).
"""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

from config.settings import XGB_PARAMS, MODELS_DIR
from src.features.first_inning import FIRST_INNING_FEATURES
from src.model.registry import save_model, load_latest_model
from src.utils.odds_math import american_to_implied
from src.utils.logging import get_logger

log = get_logger(__name__)


# ── Training ─────────────────────────────────────────────────────────────

def train_first_inning_ml(df: pd.DataFrame, train_seasons: list[int]) -> tuple:
    """Train the 1st inning moneyline model."""
    log.info("=== Training 1ST INNING MONEYLINE Model ===")
    return _train_model(
        df, train_seasons,
        feature_cols=FIRST_INNING_FEATURES,
        target_col="i1_home_won",
        model_name="xgb_i1_ml",
    )


def train_first_inning_total(df: pd.DataFrame, train_seasons: list[int]) -> tuple:
    """Train the 1st inning total (over/under 0.5) model."""
    log.info("=== Training 1ST INNING TOTAL Model ===")
    return _train_model(
        df, train_seasons,
        feature_cols=FIRST_INNING_FEATURES,
        target_col="i1_went_over",
        model_name="xgb_i1_total",
    )


def _train_model(
    df: pd.DataFrame,
    train_seasons: list[int],
    feature_cols: list,
    target_col: str,
    model_name: str,
) -> tuple:
    """Walk-forward train a 1st inning model."""
    available_cols = [c for c in feature_cols if c in df.columns]

    if target_col not in df.columns:
        log.error(f"Target column '{target_col}' not found!")
        return None, {}

    all_predictions = []
    test_seasons = [s for s in sorted(df["season"].unique()) if s >= min(train_seasons) + 2]

    for test_season in test_seasons:
        train_mask = df["season"] < test_season
        test_mask = df["season"] == test_season

        X_train = df.loc[train_mask, available_cols].fillna(0)
        y_train = df.loc[train_mask, target_col].astype(int)
        X_test = df.loc[test_mask, available_cols].fillna(0)
        y_test = df.loc[test_mask, target_col].astype(int)

        if len(X_train) < 200 or len(X_test) < 50:
            continue

        model = XGBClassifier(**XGB_PARAMS)
        model.fit(X_train, y_train, verbose=False)
        probs = model.predict_proba(X_test)[:, 1]

        preds = df.loc[test_mask, [
            "game_id", "game_date", "home_team", "away_team", "season",
            "i1_home_runs", "i1_away_runs", "i1_total",
        ]].copy()
        preds["model_prob"] = probs
        preds["model_pick"] = (probs >= 0.5).astype(int)
        preds["actual"] = y_test.values

        acc = accuracy_score(y_test, (probs >= 0.5).astype(int))
        log.info(f"  {model_name} {test_season}: {len(X_test)} games, accuracy={acc:.3f}")
        all_predictions.append(preds)

    # Final model on all data
    X_all = df[available_cols].fillna(0)
    y_all = df[target_col].astype(int)
    final_model = XGBClassifier(**XGB_PARAMS)
    final_model.fit(X_all, y_all, verbose=False)

    preds_df = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()

    if not preds_df.empty:
        overall_acc = accuracy_score(preds_df["actual"], preds_df["model_pick"])
        log.info(f"  {model_name} OVERALL: {overall_acc:.1%} on {len(preds_df)} games")
    else:
        overall_acc = 0.0

    metadata = {
        "model_type": model_name,
        "features": available_cols,
        "n_train": len(X_all),
        "test_accuracy": round(overall_acc, 4),
        "test_games": len(preds_df),
    }

    save_model(final_model, metadata, name=model_name)
    return final_model, metadata, preds_df


# ── Prediction ───────────────────────────────────────────────────────────

def predict_first_inning_ml(games: list[dict], model=None, metadata=None) -> list[dict]:
    """Generate 1st inning moneyline predictions."""
    if model is None:
        model, metadata = load_latest_model("xgb_i1_ml")
    return _predict(games, model, metadata, "1ST INN ML")


def predict_first_inning_total(games: list[dict], model=None, metadata=None) -> list[dict]:
    """Generate 1st inning total (over/under 0.5) predictions."""
    if model is None:
        model, metadata = load_latest_model("xgb_i1_total")
    return _predict(games, model, metadata, "1ST INN TOTAL")


def _predict(
    games: list[dict],
    model,
    metadata: dict,
    bet_type: str,
) -> list[dict]:
    """Generate 1st inning predictions."""
    if not games:
        return []

    trained_features = metadata.get("features", [])
    available_cols = [c for c in trained_features if c in games[0]]

    if not available_cols:
        log.warning(f"No matching features for {bet_type}")
        return []

    df = pd.DataFrame(games)
    X = df[available_cols].fillna(0)
    raw_probs = model.predict_proba(X)[:, 1]

    # Dampen toward 0.5 to reduce overconfidence
    DAMPEN = 0.35
    probs = raw_probs * (1 - DAMPEN) + 0.5 * DAMPEN

    MAX_PICKS = 2
    MIN_EDGE = 0.08

    results = []
    for i, game in enumerate(games):
        prob = float(probs[i])

        if bet_type == "1ST INN ML":
            # prob = P(home wins 1st inning)
            # For ML, we pick whichever side has higher probability
            home_prob = prob
            away_prob = 1 - prob
            home_odds = game.get("i1_home_odds", -110)
            away_odds = game.get("i1_away_odds", -110)

            # Evaluate home side
            home_implied = american_to_implied(home_odds)
            home_edge = home_prob - home_implied if home_implied else home_prob - 0.5
            # Evaluate away side
            away_implied = american_to_implied(away_odds)
            away_edge = away_prob - away_implied if away_implied else away_prob - 0.5

            if home_edge >= away_edge:
                pick_team = game.get("home_team", "")
                pick_odds = home_odds
                pick_prob = home_prob
                edge = home_edge
            else:
                pick_team = game.get("away_team", "")
                pick_odds = away_odds
                pick_prob = away_prob
                edge = away_edge

            results.append({
                "bet_type": bet_type,
                "game_date": game.get("game_date", ""),
                "home_team": game.get("home_team", ""),
                "away_team": game.get("away_team", ""),
                "home_sp_name": game.get("home_sp_name", "TBD"),
                "away_sp_name": game.get("away_sp_name", "TBD"),
                "pick": pick_team,
                "odds": pick_odds,
                "model_prob": round(pick_prob, 4),
                "edge": round(edge, 4),
                "edge_pct": f"{edge * 100:.1f}%",
                "confidence": _confidence_label(edge),
                "recommended": False,  # Set later after sorting
                "notes": _generate_ml_notes(game, pick_team, edge),
            })

        elif bet_type == "1ST INN TOTAL":
            # prob = P(over 0.5 = at least 1 run scored)
            over_prob = prob
            is_over = over_prob > 0.5
            pick_prob = over_prob if is_over else (1 - over_prob)
            edge = pick_prob - 0.5

            over_odds = game.get("i1_over_odds", -110)
            under_odds = game.get("i1_under_odds", -110)
            pick_odds = over_odds if is_over else under_odds

            # Calculate edge against market implied prob
            if is_over and over_odds:
                market_implied = american_to_implied(over_odds)
                if market_implied:
                    edge = pick_prob - market_implied
            elif not is_over and under_odds:
                market_implied = american_to_implied(under_odds)
                if market_implied:
                    edge = pick_prob - market_implied

            results.append({
                "bet_type": bet_type,
                "game_date": game.get("game_date", ""),
                "home_team": game.get("home_team", ""),
                "away_team": game.get("away_team", ""),
                "home_sp_name": game.get("home_sp_name", "TBD"),
                "away_sp_name": game.get("away_sp_name", "TBD"),
                "pick": "OVER 0.5" if is_over else "UNDER 0.5",
                "line": "0.5",
                "odds": pick_odds,
                "model_prob": round(pick_prob, 4),
                "edge": round(edge, 4),
                "edge_pct": f"{edge * 100:.1f}%",
                "confidence": _confidence_label(edge),
                "recommended": False,
                "notes": _generate_total_notes(game, is_over, edge),
            })

    # Sort by edge and cap recommendations
    results.sort(key=lambda x: x["edge"], reverse=True)
    rec_count = 0
    for r in results:
        if r["edge"] >= MIN_EDGE and rec_count < MAX_PICKS:
            r["recommended"] = True
            rec_count += 1

    return results


def _confidence_label(edge: float) -> str:
    if edge >= 0.15:
        return "HIGH"
    elif edge >= 0.12:
        return "MEDIUM"
    elif edge >= 0.08:
        return "LOW"
    else:
        return "NO PLAY"


def _generate_ml_notes(game: dict, pick_team: str, edge: float) -> str:
    notes = []
    home = game.get("home_team", "")
    is_home_pick = pick_team == home

    sp_era_combined = game.get("sp_era_combined", 4.5)
    if sp_era_combined < 3.5:
        notes.append("Elite pitching matchup")

    if is_home_pick:
        home_1st = game.get("home_1st_inn_scored_pct", 0.3)
        if home_1st > 0.4:
            notes.append("Home team scores 1st often")
    else:
        away_1st = game.get("away_1st_inn_scored_pct", 0.3)
        if away_1st > 0.4:
            notes.append("Away team scores 1st often")

    if edge >= 0.10:
        notes.append("Strong value")

    return "; ".join(notes) if notes else "Standard play"


def _generate_total_notes(game: dict, is_over: bool, edge: float) -> str:
    notes = []

    combined_1st = game.get("combined_1st_inn_rpg", 0.8)
    if is_over:
        if combined_1st > 1.0:
            notes.append("Both teams score early")
        pf = game.get("park_factor", 1.0)
        if pf >= 1.04:
            notes.append("Hitter-friendly park")
    else:
        if combined_1st < 0.6:
            notes.append("Low 1st inning scoring")
        sp_era = game.get("sp_era_combined", 4.5)
        if sp_era < 3.5:
            notes.append("Elite SPs keep 1st clean")

    if edge >= 0.10:
        notes.append("Strong value")

    return "; ".join(notes) if notes else "Standard play"
