#!/usr/bin/env python3
"""
Run the full backtesting pipeline.
Usage: python run_backtest.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.backtest.runner import run_full_backtest
from src.model.evaluate import print_report
from src.utils.logging import get_logger

log = get_logger("backtest")


def main():
    log.info("Starting full backtest...")
    metrics = run_full_backtest()

    if "error" in metrics:
        log.error(f"Backtest failed: {metrics['error']}")
        sys.exit(1)

    report = print_report(metrics)
    print(report)

    # Summary for quick reference
    print(f"\nKey Results:")
    print(f"  Hit Rate: {metrics.get('hit_rate', 0)}%")
    print(f"  ROI: {metrics.get('roi_pct', 0):.1f}%")
    if "filtered_hit_rate" in metrics:
        print(f"  Filtered Hit Rate (edge>threshold): {metrics['filtered_hit_rate']:.1%}")
        print(f"  Filtered ROI: {metrics.get('filtered_roi_pct', 0):.1f}%")


if __name__ == "__main__":
    main()
