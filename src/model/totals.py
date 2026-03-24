"""
Totals model training and prediction for MLB.
Two models: Full Game Over/Under and First 5 Innings Over/Under.
"""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, log_loss

from config.settings import XGB_PARAMS, MODELS_DIR
from src.features.totals import TOTALS_FEATURE_COLUMNS, F5_FEATURE_COLUMNS
from src.model.registry import save_model, load_latest_model
from src.utils.odds_math import american_to_implied
from src.utils.logging import get_logger

log = get_logger(__name__)


# ── Training ─────────────────────────────────────────────────────────────

def train_full_game_total(df: pd.DataFrame, train_seasons: list[int]) -> tuple:
    """Train the full-game totals model using walk-forward validation."""
    log.info("=== Training FULL GAME TOTALS Model ===")
    return _train_totals_model(
        df, train_seasons,
        feature_cols=TOTALS_FEATURE_COLUMNS,
        target_col="fg_went_over",
        model_name="xgb_total_fg",
    )


def train_f5_total(df: pd.DataFrame, train_seasons: list[int]) -> tuple:
    """Train the first-5-innings totals model using walk-forward validation."""
    log.info("=== Training FIRST 5 INNINGS TOTALS Model ===")
    return _train_totals_model(
        df, train_seasons,
        feature_cols=F5_FEATURE_COLUMNS,
        model_name="xgb_total_f5",
        target_col="f5_went_over",
    )


def _train_totals_model(
    df: pd.DataFrame,
    train_seasons: list[int],
    feature_cols: list,
    target_col: str,
    model_name: str,
) -> tuple:
    """Walk-forward train a totals model."""
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
            "total_runs", "f5_total",
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

    # Overall accuracy
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

def predict_full_game_totals(games: list[dict], model=None, metadata=None) -> pd.DataFrame:
    """Generate full-game over/under predictions."""
    if model is None:
        model, metadata = load_latest_model("xgb_total_fg")
    return _predict_totals(games, model, metadata, "FULL GAME TOTAL")


def predict_f5_totals(games: list[dict], model=None, metadata=None) -> pd.DataFrame:
    """Generate F5 innings over/under predictions."""
    if model is None:
        model, metadata = load_latest_model("xgb_total_f5")
    return _predict_totals(games, model, metadata, "F5 TOTAL")


def _predict_totals(
    games: list[dict],
    model,
    metadata: dict,
    bet_type: str,
) -> pd.DataFrame:
    """Generate totals predictions from a trained model."""
    if not games:
        return pd.DataFrame()

    trained_features = metadata.get("features", [])
    available_cols = [c for c in trained_features if c in games[0]]

    if not available_cols:
        log.warning(f"No matching features for {bet_type}")
        return pd.DataFrame()

    df = pd.DataFrame(games)
    X = df[available_cols].fillna(0)
    probs = model.predict_proba(X)[:, 1]  # P(over)

    results = []
    for i, game in enumerate(games):
        over_prob = float(probs[i])
        is_over = over_prob > 0.5
        pick_prob = over_prob if is_over else (1 - over_prob)
        edge = pick_prob - 0.5

        total_line = game.get("total_line", 0)
        if bet_type == "F5 TOTAL" and total_line > 0:
            # F5 line is typically ~55% of full game line
            f5_line = game.get("f5_line", round(total_line * 0.55, 1))
            line_display = str(f5_line)
        else:
            line_display = str(total_line)

        results.append({
            "bet_type": bet_type,
            "game_date": game.get("game_date", ""),
            "home_team": game.get("home_team", ""),
            "away_team": game.get("away_team", ""),
            "home_sp_name": game.get("home_sp_name", "TBD"),
            "away_sp_name": game.get("away_sp_name", "TBD"),
            "pick": "OVER" if is_over else "UNDER",
            "line": line_display,
            "model_prob": round(pick_prob, 4),
            "edge": round(edge, 4),
            "edge_pct": f"{edge * 100:.1f}%",
            "confidence": _confidence_label(edge),
            "recommended": edge >= 0.03,
            "notes": _generate_notes(game, over_prob, edge, bet_type),
        })

    return pd.DataFrame(results).sort_values("edge", ascending=False)


def _confidence_label(edge: float) -> str:
    if edge >= 0.08:
        return "HIGH"
    elif edge >= 0.05:
        return "MEDIUM"
    elif edge >= 0.03:
        return "LOW"
    else:
        return "NO PLAY"


def _generate_notes(game: dict, over_prob: float, edge: float, bet_type: str) -> str:
    notes = []
    pf = game.get("park_factor", 1.0)
    if pf >= 1.04:
        notes.append("Hitter-friendly park")
    elif pf <= 0.96:
        notes.append("Pitcher-friendly park")

    combined_rpg = game.get("combined_rpg", 9.0)
    if combined_rpg > 10:
        notes.append("High-scoring matchup")
    elif combined_rpg < 7.5:
        notes.append("Low-scoring matchup")

    sp_era = game.get("sp_era_combined", 4.5)
    if sp_era < 3.5:
        notes.append("Strong SP matchup")
    elif sp_era > 5.0:
        notes.append("Weak SP matchup")

    if edge >= 0.08:
        notes.append("Strong value")

    return "; ".join(notes) if notes else "Standard play"
