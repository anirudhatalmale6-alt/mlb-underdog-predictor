#!/usr/bin/env python3
"""
Run the daily prediction pipeline.
Usage: python run_daily.py [YYYY-MM-DD]
"""

import sys
import os
from datetime import date, datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipeline.daily import run_daily_pipeline
from src.utils.logging import get_logger

log = get_logger("daily")


def main():
    # Optional date argument
    target_date = date.today()
    if len(sys.argv) > 1:
        try:
            target_date = datetime.strptime(sys.argv[1], "%Y-%m-%d").date()
        except ValueError:
            print(f"Invalid date format: {sys.argv[1]}. Use YYYY-MM-DD.")
            sys.exit(1)

    log.info(f"Running daily pipeline for {target_date}...")
    predictions = run_daily_pipeline(target_date)

    if predictions.empty:
        print(f"\nNo picks for {target_date}.")
        return

    # Print picks to console
    recommended = predictions[predictions["recommended"]]
    print(f"\n{'='*60}")
    print(f"  MLB UNDERDOG PICKS — {target_date}")
    print(f"{'='*60}")

    if recommended.empty:
        print("  No recommended plays today (no picks met edge threshold).")
    else:
        print(f"  {len(recommended)} RECOMMENDED PLAY(S):\n")
        for _, pick in recommended.iterrows():
            print(f"  {pick['underdog_team']:>4s} ({pick['underdog_odds']:+d})")
            print(f"       vs {pick['home_team']} @ {pick['away_team']}")
            print(f"       Win Prob: {pick['model_win_prob']:.1%} | "
                  f"Edge: {pick['edge_pct']} | "
                  f"Confidence: {pick['confidence']}")
            print(f"       {pick['notes']}")
            print()

    # Also show non-recommended for awareness
    not_rec = predictions[~predictions["recommended"]]
    if not not_rec.empty:
        print(f"  {len(not_rec)} other qualifying underdogs (below edge threshold):")
        for _, pick in not_rec.iterrows():
            print(f"    {pick['underdog_team']:>4s} ({pick['underdog_odds']:+d}) | "
                  f"Prob: {pick['model_win_prob']:.1%} | Edge: {pick['edge_pct']}")

    print(f"\n{'='*60}")
    print(f"  Output saved to data/output/picks_{target_date}.csv")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
