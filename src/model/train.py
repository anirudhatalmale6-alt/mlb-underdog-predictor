"""
Model training with XGBoost and walk-forward validation.
"""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss, brier_score_loss, accuracy_score
import optuna

from config.settings import XGB_PARAMS, MIN_UNDERDOG_ODDS, MAX_UNDERDOG_ODDS
from src.features.builder import FEATURE_COLUMNS
from src.utils.logging import get_logger

log = get_logger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def train_model(
    features_df: pd.DataFrame,
    train_seasons: list[int] = None,
    tune_hyperparams: bool = False,
) -> tuple[XGBClassifier, dict]:
    """
    Train an XGBoost model on historical data.

    Args:
        features_df: DataFrame with features + underdog_won column
        train_seasons: Which seasons to train on (default: all available)
        tune_hyperparams: Whether to run Optuna hyperparameter tuning

    Returns:
        (trained model, metadata dict with metrics)
    """
    df = features_df.copy()

    # Filter to qualifying underdogs
    df = df[
        (df["underdog_odds"] >= MIN_UNDERDOG_ODDS) &
        (df["underdog_odds"] <= MAX_UNDERDOG_ODDS)
    ].copy()

    if train_seasons:
        df = df[df["game_date"].dt.year.isin(train_seasons)]

    if len(df) < 100:
        log.warning(f"Only {len(df)} training samples. Model may be unreliable.")

    # Prepare features and target
    available_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    X = df[available_cols].copy()
    y = df["underdog_won"].astype(int)

    # Fill missing values with column median
    X = X.fillna(X.median())

    log.info(f"Training on {len(X)} games, {len(available_cols)} features")
    log.info(f"Underdog win rate in training data: {y.mean():.3f}")

    if tune_hyperparams and len(X) > 500:
        params = _tune_hyperparams(X, y)
    else:
        params = XGB_PARAMS.copy()

    model = XGBClassifier(**params)
    model.fit(X, y, verbose=False)

    # Training metrics
    train_probs = model.predict_proba(X)[:, 1]
    train_preds = (train_probs >= 0.5).astype(int)

    metadata = {
        "n_train": len(X),
        "features": available_cols,
        "underdog_win_rate": float(y.mean()),
        "train_accuracy": float(accuracy_score(y, train_preds)),
        "train_log_loss": float(log_loss(y, train_probs)),
        "train_brier": float(brier_score_loss(y, train_probs)),
        "params": params,
    }

    log.info(f"Training accuracy: {metadata['train_accuracy']:.3f}")
    log.info(f"Training log loss: {metadata['train_log_loss']:.4f}")

    return model, metadata


def walk_forward_validation(
    features_df: pd.DataFrame,
    test_seasons: list[int],
) -> tuple[XGBClassifier, pd.DataFrame]:
    """
    Walk-forward validation: for each test season, train on all prior seasons.

    Returns:
        (final model trained on all data, DataFrame of out-of-sample predictions)
    """
    df = features_df.copy()
    df = df[
        (df["underdog_odds"] >= MIN_UNDERDOG_ODDS) &
        (df["underdog_odds"] <= MAX_UNDERDOG_ODDS)
    ].copy()

    available_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    all_predictions = []

    for test_season in test_seasons:
        train_mask = df["game_date"].dt.year < test_season
        test_mask = df["game_date"].dt.year == test_season

        X_train = df.loc[train_mask, available_cols].fillna(0)
        y_train = df.loc[train_mask, "underdog_won"].astype(int)
        X_test = df.loc[test_mask, available_cols].fillna(0)
        y_test = df.loc[test_mask, "underdog_won"].astype(int)

        if len(X_train) < 50 or len(X_test) < 10:
            log.warning(f"Skipping season {test_season}: insufficient data "
                       f"(train={len(X_train)}, test={len(X_test)})")
            continue

        model = XGBClassifier(**XGB_PARAMS)
        model.fit(X_train, y_train, verbose=False)

        probs = model.predict_proba(X_test)[:, 1]

        season_preds = df.loc[test_mask, [
            "game_id", "game_date", "home_team", "away_team",
            "underdog_team", "underdog_odds", "underdog_won"
        ]].copy()
        season_preds["model_prob"] = probs
        season_preds["model_pick"] = (probs >= 0.5).astype(int)
        season_preds["test_season"] = test_season

        acc = accuracy_score(y_test, (probs >= 0.5).astype(int))
        log.info(f"Season {test_season}: {len(X_test)} games, "
                f"accuracy={acc:.3f}, underdog win rate={y_test.mean():.3f}")

        all_predictions.append(season_preds)

    # Train final model on all available data
    X_all = df[available_cols].fillna(0)
    y_all = df["underdog_won"].astype(int)
    final_model = XGBClassifier(**XGB_PARAMS)
    final_model.fit(X_all, y_all, verbose=False)

    predictions_df = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    return final_model, predictions_df


def _tune_hyperparams(X: pd.DataFrame, y: pd.Series, n_trials: int = 30) -> dict:
    """Use Optuna to tune XGBoost hyperparameters."""
    log.info(f"Tuning hyperparameters with {n_trials} trials...")

    def objective(trial):
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 7),
            "subsample": trial.suggest_float("subsample", 0.6, 0.9),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": 0,
        }

        # Time-series cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        scores = []
        for train_idx, val_idx in tscv.split(X):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = XGBClassifier(**params)
            model.fit(X_tr, y_tr, verbose=False)
            probs = model.predict_proba(X_val)[:, 1]
            scores.append(log_loss(y_val, probs))

        return np.mean(scores)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    best.update({
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": 0,
    })

    log.info(f"Best params: {best}")
    return best
