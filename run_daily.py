#!/usr/bin/env python3
"""
Run the daily prediction pipeline.
Usage: python run_daily.py [YYYY-MM-DD]
"""

import sys
import os
from datetime import date, datetime
import pandas as pd

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
    results = run_daily_pipeline(target_date)

    # results is a dict: {'moneyline': df, 'full_game_total': df, 'f5_total': df}
    has_any = any(not df.empty for df in results.values())
    if not has_any:
        print(f"\nNo picks for {target_date}.")
        return

    print(f"\n{'='*60}")
    print(f"  MLB PICKS — {target_date}")
    print(f"{'='*60}")

    # ── Moneyline Underdogs ──
    ml = results.get("moneyline", pd.DataFrame())
    if not ml.empty:
        print(f"\n  MONEYLINE UNDERDOGS")
        print(f"  {'─'*40}")
        recommended = ml[ml["recommended"]]
        if recommended.empty:
            print("  No recommended underdog plays today.")
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

        not_rec = ml[~ml["recommended"]]
        if not not_rec.empty:
            print(f"  {len(not_rec)} other qualifying underdogs (below edge threshold):")
            for _, pick in not_rec.iterrows():
                print(f"    {pick['underdog_team']:>4s} ({pick['underdog_odds']:+d}) | "
                      f"Prob: {pick['model_win_prob']:.1%} | Edge: {pick['edge_pct']}")

    # ── Totals (Full Game + F5) ──
    for key, label in [("full_game_total", "FULL GAME TOTALS"), ("f5_total", "FIRST 5 INNINGS TOTALS")]:
        df = results.get(key, pd.DataFrame())
        if df.empty:
            continue
        print(f"\n  {label}")
        print(f"  {'─'*40}")
        rec = df[df["recommended"]] if "recommended" in df.columns else pd.DataFrame()
        if rec.empty:
            print("  No recommended totals plays today.")
        else:
            print(f"  {len(rec)} RECOMMENDED PLAY(S):\n")
            for _, pick in rec.iterrows():
                print(f"  {pick.get('away_team', '?')} @ {pick.get('home_team', '?')}")
                print(f"       {pick.get('pick', '?')} {pick.get('line', '?')} | "
                      f"Prob: {pick.get('model_prob', 0):.1%} | "
                      f"Edge: {pick.get('edge_pct', '')} | "
                      f"Confidence: {pick.get('confidence', '')}")
                notes = pick.get('notes', '')
                if notes:
                    print(f"       {notes}")
                print()

    # ── Player Props ──
    for key, label in [("pitcher_k", "PITCHER STRIKEOUT PROPS"), ("batter_hits", "BATTER HITS PROPS")]:
        df = results.get(key, pd.DataFrame())
        if df.empty:
            continue
        print(f"\n  {label}")
        print(f"  {'─'*40}")
        rec = df[df["recommended"]] if "recommended" in df.columns else pd.DataFrame()
        if rec.empty:
            print("  No recommended props plays today.")
        else:
            print(f"  {len(rec)} RECOMMENDED PLAY(S):\n")
            for _, pick in rec.iterrows():
                print(f"  {pick.get('player_name', '?')}")
                print(f"       {pick.get('pick', '?')} ({pick.get('odds', '')})")
                print(f"       {pick.get('away_team', '?')} @ {pick.get('home_team', '?')}")
                print(f"       Prob: {pick.get('model_prob', 0):.1%} | "
                      f"Edge: {pick.get('edge_pct', '')} | "
                      f"Confidence: {pick.get('confidence', '')}")
                notes = pick.get('notes', '')
                if notes:
                    print(f"       {notes}")
                print()

    print(f"\n{'='*60}")
    print(f"  Output saved to data/output/picks_{target_date}.csv")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
