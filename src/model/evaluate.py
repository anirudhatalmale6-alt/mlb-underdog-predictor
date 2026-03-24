"""
Model evaluation — ROI simulation, calibration, and performance metrics.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, brier_score_loss, accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config.settings import EDGE_THRESHOLD, OUTPUT_DIR
from src.utils.odds_math import odds_to_decimal
from src.utils.logging import get_logger

log = get_logger(__name__)


def evaluate_predictions(predictions_df: pd.DataFrame, output_dir=None) -> dict:
    """
    Comprehensive evaluation of model predictions.

    Args:
        predictions_df: DataFrame with model_prob, underdog_won, underdog_odds columns

    Returns:
        Dict of evaluation metrics
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR

    df = predictions_df.copy()
    if df.empty:
        return {"error": "No predictions to evaluate"}

    y_true = df["underdog_won"].astype(int)
    y_prob = df["model_prob"]
    y_pred = (y_prob >= 0.5).astype(int)

    # Basic metrics
    metrics = {
        "total_games": len(df),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "log_loss": float(log_loss(y_true, y_prob)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "actual_underdog_win_rate": float(y_true.mean()),
        "predicted_underdog_win_rate": float(y_prob.mean()),
    }

    # ROI simulation: flat $100 bet on all model picks
    roi_results = _simulate_roi(df)
    metrics.update(roi_results)

    # ROI with edge threshold
    edge_df = df.copy()
    if "implied_prob_market" not in edge_df.columns:
        from src.utils.odds_math import american_to_implied
        edge_df["implied_prob_market"] = edge_df["underdog_odds"].apply(
            lambda x: american_to_implied(int(x))
        )
    edge_df["edge"] = edge_df["model_prob"] - edge_df["implied_prob_market"]
    filtered = edge_df[edge_df["edge"] >= EDGE_THRESHOLD]
    if len(filtered) > 0:
        filtered_roi = _simulate_roi(filtered)
        metrics["filtered_total_picks"] = len(filtered)
        metrics["filtered_hit_rate"] = float(filtered["underdog_won"].mean())
        metrics["filtered_roi_pct"] = filtered_roi["roi_pct"]
        metrics["filtered_total_profit"] = filtered_roi["total_profit"]

    # Per-season breakdown
    if "test_season" in df.columns:
        season_metrics = {}
        for season in sorted(df["test_season"].unique()):
            sdf = df[df["test_season"] == season]
            sy = sdf["underdog_won"].astype(int)
            sp = sdf["model_prob"]
            sroi = _simulate_roi(sdf)
            season_metrics[int(season)] = {
                "games": len(sdf),
                "accuracy": float(accuracy_score(sy, (sp >= 0.5).astype(int))),
                "hit_rate": float(sy.mean()),
                "roi_pct": sroi["roi_pct"],
            }
        metrics["per_season"] = season_metrics

    # Odds bucket analysis
    buckets = [(130, 150), (150, 180), (180, 210), (210, 250)]
    bucket_analysis = {}
    for lo, hi in buckets:
        bdf = df[(df["underdog_odds"] >= lo) & (df["underdog_odds"] < hi)]
        if len(bdf) > 0:
            bucket_analysis[f"+{lo}_to_+{hi}"] = {
                "games": len(bdf),
                "hit_rate": float(bdf["underdog_won"].mean()),
                "model_accuracy": float(
                    accuracy_score(
                        bdf["underdog_won"].astype(int),
                        (bdf["model_prob"] >= 0.5).astype(int)
                    )
                ),
            }
    metrics["odds_buckets"] = bucket_analysis

    # Generate charts
    _generate_charts(df, output_dir)

    return metrics


def _simulate_roi(df: pd.DataFrame, bet_size: float = 100.0) -> dict:
    """Simulate flat-betting ROI."""
    total_wagered = 0
    total_returned = 0

    for _, row in df.iterrows():
        odds = int(row["underdog_odds"])
        won = int(row["underdog_won"])
        decimal = odds_to_decimal(odds)

        total_wagered += bet_size
        if won:
            total_returned += bet_size * decimal

    profit = total_returned - total_wagered
    roi = (profit / total_wagered * 100) if total_wagered > 0 else 0

    return {
        "total_wagered": round(total_wagered, 2),
        "total_returned": round(total_returned, 2),
        "total_profit": round(profit, 2),
        "roi_pct": round(roi, 2),
        "total_bets": len(df),
        "wins": int(df["underdog_won"].sum()),
        "hit_rate": round(float(df["underdog_won"].mean()) * 100, 1),
    }


def _generate_charts(df: pd.DataFrame, output_dir) -> None:
    """Generate evaluation charts."""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Calibration plot
        ax = axes[0, 0]
        _calibration_plot(df, ax)

        # 2. Cumulative profit curve
        ax = axes[0, 1]
        _cumulative_profit_plot(df, ax)

        # 3. Hit rate by odds bucket
        ax = axes[1, 0]
        _odds_bucket_plot(df, ax)

        # 4. Prediction distribution
        ax = axes[1, 1]
        _prediction_distribution(df, ax)

        plt.tight_layout()
        chart_path = output_dir / "backtest_report.png"
        plt.savefig(chart_path, dpi=150, bbox_inches="tight")
        plt.close()
        log.info(f"Charts saved to {chart_path}")
    except Exception as e:
        log.warning(f"Failed to generate charts: {e}")


def _calibration_plot(df, ax):
    """Plot predicted probability vs actual win rate."""
    bins = np.linspace(0, 1, 11)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    actual_rates = []

    for i in range(len(bins) - 1):
        mask = (df["model_prob"] >= bins[i]) & (df["model_prob"] < bins[i + 1])
        subset = df[mask]
        if len(subset) > 0:
            actual_rates.append(subset["underdog_won"].mean())
        else:
            actual_rates.append(np.nan)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
    ax.plot(bin_centers, actual_rates, "bo-", label="Model")
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Actual Win Rate")
    ax.set_title("Calibration Plot")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def _cumulative_profit_plot(df, ax):
    """Plot cumulative profit over time."""
    df_sorted = df.sort_values("game_date")
    profits = []
    cumulative = 0
    for _, row in df_sorted.iterrows():
        odds = int(row["underdog_odds"])
        decimal = odds_to_decimal(odds)
        if row["underdog_won"]:
            cumulative += 100 * (decimal - 1)
        else:
            cumulative -= 100
        profits.append(cumulative)

    ax.plot(range(len(profits)), profits, "b-", linewidth=0.8)
    ax.axhline(y=0, color="r", linestyle="--", alpha=0.5)
    ax.set_xlabel("Bet Number")
    ax.set_ylabel("Cumulative Profit ($)")
    ax.set_title("Cumulative Profit (Flat $100 Bets)")
    ax.fill_between(range(len(profits)), profits, 0,
                    where=[p > 0 for p in profits], alpha=0.2, color="green")
    ax.fill_between(range(len(profits)), profits, 0,
                    where=[p <= 0 for p in profits], alpha=0.2, color="red")


def _odds_bucket_plot(df, ax):
    """Plot hit rate by odds bucket."""
    buckets = [(130, 150), (150, 180), (180, 210), (210, 250)]
    labels = []
    hit_rates = []
    counts = []

    for lo, hi in buckets:
        mask = (df["underdog_odds"] >= lo) & (df["underdog_odds"] < hi)
        subset = df[mask]
        labels.append(f"+{lo}\nto +{hi}")
        hit_rates.append(subset["underdog_won"].mean() * 100 if len(subset) > 0 else 0)
        counts.append(len(subset))

    bars = ax.bar(labels, hit_rates, color="steelblue", alpha=0.8)
    ax.set_ylabel("Win Rate (%)")
    ax.set_title("Underdog Win Rate by Odds Range")

    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"n={count}", ha="center", va="bottom", fontsize=9)


def _prediction_distribution(df, ax):
    """Plot distribution of model predictions."""
    wins = df[df["underdog_won"] == 1]["model_prob"]
    losses = df[df["underdog_won"] == 0]["model_prob"]

    ax.hist(losses, bins=30, alpha=0.5, label="Losses", color="red")
    ax.hist(wins, bins=30, alpha=0.5, label="Wins", color="green")
    ax.set_xlabel("Model Predicted Probability")
    ax.set_ylabel("Count")
    ax.set_title("Prediction Distribution")
    ax.legend()


def print_report(metrics: dict) -> str:
    """Format metrics as a readable report string."""
    lines = [
        "=" * 60,
        "  MLB UNDERDOG PREDICTOR — BACKTEST REPORT",
        "=" * 60,
        "",
        f"  Total games evaluated:     {metrics.get('total_games', 0)}",
        f"  Model accuracy:            {metrics.get('accuracy', 0):.1%}",
        f"  Actual underdog win rate:  {metrics.get('actual_underdog_win_rate', 0):.1%}",
        f"  Brier score:               {metrics.get('brier_score', 0):.4f}",
        "",
        "  ── ROI (Flat $100 Bets on All Qualifying Underdogs) ──",
        f"  Total bets:      {metrics.get('total_bets', 0)}",
        f"  Wins:            {metrics.get('wins', 0)}",
        f"  Hit rate:        {metrics.get('hit_rate', 0)}%",
        f"  Total wagered:   ${metrics.get('total_wagered', 0):,.2f}",
        f"  Total returned:  ${metrics.get('total_returned', 0):,.2f}",
        f"  Profit/Loss:     ${metrics.get('total_profit', 0):,.2f}",
        f"  ROI:             {metrics.get('roi_pct', 0):.1f}%",
        "",
    ]

    if "filtered_total_picks" in metrics:
        lines.extend([
            f"  ── ROI (Edge >= {EDGE_THRESHOLD*100:.0f}% Threshold Only) ──",
            f"  Total picks:     {metrics.get('filtered_total_picks', 0)}",
            f"  Hit rate:        {metrics.get('filtered_hit_rate', 0):.1%}",
            f"  ROI:             {metrics.get('filtered_roi_pct', 0):.1f}%",
            f"  Profit/Loss:     ${metrics.get('filtered_total_profit', 0):,.2f}",
            "",
        ])

    if "per_season" in metrics:
        lines.append("  ── Per-Season Breakdown ──")
        for season, sm in metrics["per_season"].items():
            lines.append(
                f"  {season}: {sm['games']} games, "
                f"accuracy={sm['accuracy']:.1%}, "
                f"hit_rate={sm['hit_rate']:.1%}, "
                f"ROI={sm['roi_pct']:.1f}%"
            )
        lines.append("")

    if "odds_buckets" in metrics:
        lines.append("  ── Performance by Odds Range ──")
        for bucket, bm in metrics["odds_buckets"].items():
            lines.append(
                f"  {bucket}: {bm['games']} games, "
                f"hit_rate={bm['hit_rate']:.1%}, "
                f"model_acc={bm['model_accuracy']:.1%}"
            )

    lines.append("=" * 60)
    report = "\n".join(lines)
    return report
