"""
Backtest runner — orchestrates historical data collection, feature building,
model training, and evaluation.
"""

import json
import pandas as pd

from config.settings import (
    HISTORICAL_SEASONS, BACKTEST_TEST_SEASONS,
    PROCESSED_DIR, OUTPUT_DIR,
)
from src.ingest.historical import build_historical_dataset
from src.features.builder import build_historical_features
from src.model.train import walk_forward_validation
from src.model.evaluate import evaluate_predictions, print_report
from src.model.registry import save_model
from src.utils.logging import get_logger

log = get_logger(__name__)


def run_full_backtest(
    seasons: list[int] = None,
    test_seasons: list[int] = None,
) -> dict:
    """
    Run the complete backtesting pipeline:
    1. Collect historical game data
    2. Build features
    3. Walk-forward train/test
    4. Evaluate and save results

    Returns:
        Evaluation metrics dict
    """
    if seasons is None:
        seasons = HISTORICAL_SEASONS
    if test_seasons is None:
        test_seasons = BACKTEST_TEST_SEASONS

    # Step 1: Collect historical games
    log.info("Step 1: Collecting historical game data...")
    games_df = build_historical_dataset(seasons)
    if games_df.empty:
        log.error("No historical data collected. Aborting backtest.")
        return {"error": "No data"}

    log.info(f"  Collected {len(games_df)} games across seasons {seasons}")

    # Step 2: Build features
    log.info("Step 2: Building feature matrix...")
    features_df = build_historical_features(games_df)
    if features_df.empty:
        log.error("No features built. Aborting backtest.")
        return {"error": "No features"}

    # Save features
    features_path = PROCESSED_DIR / "features.parquet"
    features_df.to_parquet(features_path, index=False)
    log.info(f"  Built {len(features_df)} feature vectors, saved to {features_path}")

    # Step 3: Walk-forward validation
    log.info(f"Step 3: Walk-forward validation (test seasons: {test_seasons})...")
    final_model, predictions_df = walk_forward_validation(features_df, test_seasons)

    if predictions_df.empty:
        log.error("No predictions generated. Aborting backtest.")
        return {"error": "No predictions"}

    log.info(f"  Generated {len(predictions_df)} out-of-sample predictions")

    # Step 4: Evaluate
    log.info("Step 4: Evaluating model performance...")
    metrics = evaluate_predictions(predictions_df)
    report = print_report(metrics)
    log.info("\n" + report)

    # Save results
    predictions_path = OUTPUT_DIR / "backtest_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)

    metrics_path = OUTPUT_DIR / "backtest_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    report_path = OUTPUT_DIR / "backtest_report.txt"
    with open(report_path, "w") as f:
        f.write(report)

    # Save the final model
    save_model(final_model, {
        "trained_on": str(seasons),
        "test_seasons": str(test_seasons),
        "n_features": len(features_df.columns) - 6,  # Subtract metadata cols
        "backtest_metrics": metrics,
    })

    log.info("Backtest complete! Results saved to data/output/")
    return metrics
