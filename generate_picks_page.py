#!/usr/bin/env python3
"""
Generate a clean, readable picks page (TODAYS_PICKS.md) for easy viewing on GitHub.
This runs after the daily pipeline and formats the output for non-technical users.
"""

import sys
import os
import json
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import OUTPUT_DIR

PROJECT_ROOT = Path(__file__).resolve().parent


def generate_picks_page(target_date: date = None):
    if target_date is None:
        target_date = date.today()

    date_str = target_date.strftime("%Y-%m-%d")
    json_path = OUTPUT_DIR / f"picks_{date_str}.json"

    output_path = PROJECT_ROOT / "TODAYS_PICKS.md"

    # Header
    lines = []
    lines.append(f"# MLB Picks - {target_date.strftime('%A, %B %d, %Y')}")
    lines.append("")
    lines.append(f"*Last updated: {datetime.now().strftime('%Y-%m-%d %I:%M %p ET')}*")
    lines.append("")

    if not json_path.exists():
        lines.append("## No Games Today")
        lines.append("")
        lines.append("No regular-season MLB games scheduled today.")
        lines.append("")
        lines.append("---")
        lines.append("*This page updates automatically every morning during the MLB season.*")
        with open(output_path, "w") as f:
            f.write("\n".join(lines))
        print(f"No games today. Wrote {output_path}")
        return

    with open(json_path) as f:
        picks = json.load(f)

    if not picks:
        lines.append("## No Picks Today")
        lines.append("")
        lines.append("There are games today but no picks met the model's criteria.")
        lines.append("")
        lines.append("---")
        lines.append("*This page updates automatically every morning during the MLB season.*")
        with open(output_path, "w") as f:
            f.write("\n".join(lines))
        print(f"No picks. Wrote {output_path}")
        return

    # Separate by bet type
    ml_picks = [p for p in picks if p.get("bet_type", "MONEYLINE") == "MONEYLINE"]
    fg_picks = [p for p in picks if p.get("bet_type") == "FULL GAME TOTAL"]
    f5_picks = [p for p in picks if p.get("bet_type") == "F5 TOTAL"]

    # If no bet_type field (legacy), treat all as moneyline
    if not ml_picks and not fg_picks and not f5_picks:
        ml_picks = picks

    # ── Moneyline Underdog Section ──
    if ml_picks:
        _render_moneyline_section(lines, ml_picks)

    # ── Full Game Totals Section ──
    if fg_picks:
        _render_totals_section(lines, fg_picks, "Full Game Over/Under")

    # ── F5 Totals Section ──
    if f5_picks:
        _render_totals_section(lines, f5_picks, "First 5 Innings Over/Under")

    if not ml_picks and not fg_picks and not f5_picks:
        lines.append("## No Recommended Plays Today")
        lines.append("")

    # Footer
    lines.append("---")
    lines.append("")
    lines.append("### How to Read This")
    lines.append("")
    lines.append("**Moneyline Underdogs:**")
    lines.append("- **Odds**: The moneyline payout (e.g., +150 means $100 bet wins $150)")
    lines.append("- **Win Probability**: The model's estimated chance of this underdog winning")
    lines.append("- **Edge**: How much higher the model's probability is vs what the odds imply (positive = value bet)")
    lines.append("")
    lines.append("**Totals (Over/Under):**")
    lines.append("- **Line**: The sportsbook's projected total runs for the game")
    lines.append("- **Pick**: OVER or UNDER — the model's prediction")
    lines.append("- **Probability**: The model's confidence in its pick")
    lines.append("- **Edge**: How much the model disagrees with the 50/50 line (higher = more confident)")
    lines.append("")
    lines.append("**Confidence Levels:** HIGH (8%+ edge), MEDIUM (5-8%), LOW (3-5%)")
    lines.append("")
    lines.append("*Disclaimer: This is a statistical model for informational purposes. Past performance does not guarantee future results. Gamble responsibly.*")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    total = len(ml_picks) + len(fg_picks) + len(f5_picks)
    print(f"Wrote {total} picks ({len(ml_picks)} ML, {len(fg_picks)} FG totals, {len(f5_picks)} F5 totals) to {output_path}")


def _render_moneyline_section(lines: list, picks: list):
    """Render the moneyline underdog picks section."""
    recommended = [p for p in picks if p.get("recommended")]
    others = [p for p in picks if not p.get("recommended")]

    lines.append("## Moneyline Underdogs")
    lines.append("")

    if recommended:
        lines.append(f"### RECOMMENDED PLAYS ({len(recommended)})")
        lines.append("")
        lines.append("These picks have a positive edge over the market odds:")
        lines.append("")

        for p in recommended:
            team = p.get("underdog_team", "???")
            odds = p.get("underdog_odds", 0)
            prob = p.get("model_win_prob", 0)
            edge = p.get("edge_pct", "0.0%")
            conf = p.get("confidence", "")
            home = p.get("home_team", "")
            away = p.get("away_team", "")
            home_sp = p.get("home_sp_name", "TBD")
            away_sp = p.get("away_sp_name", "TBD")
            notes = p.get("notes", "")

            odds_str = f"+{odds}" if odds > 0 else str(odds)
            prob_str = f"{prob:.1%}" if isinstance(prob, float) else str(prob)

            lines.append(f"#### {team} ({odds_str})")
            lines.append("")
            lines.append(f"| Stat | Value |")
            lines.append(f"|------|-------|")
            lines.append(f"| Matchup | {away} @ {home} |")
            lines.append(f"| Starting Pitchers | {away_sp} vs {home_sp} |")
            lines.append(f"| Model Win Probability | {prob_str} |")
            lines.append(f"| Edge Over Market | {edge} |")
            lines.append(f"| Confidence | {conf} |")
            if notes:
                lines.append(f"| Notes | {notes} |")
            lines.append("")
    else:
        lines.append("No recommended underdog plays today.")
        lines.append("")

    if others:
        lines.append(f"**Other Qualifying Underdogs ({len(others)})** — below edge threshold:")
        lines.append("")
        lines.append("| Team | Odds | Win Prob | Edge |")
        lines.append("|------|------|----------|------|")
        for p in others:
            team = p.get("underdog_team", "???")
            odds = p.get("underdog_odds", 0)
            prob = p.get("model_win_prob", 0)
            edge = p.get("edge_pct", "0.0%")
            odds_str = f"+{odds}" if odds > 0 else str(odds)
            prob_str = f"{prob:.1%}" if isinstance(prob, float) else str(prob)
            lines.append(f"| {team} | {odds_str} | {prob_str} | {edge} |")
        lines.append("")

    lines.append("---")
    lines.append("")


def _render_totals_section(lines: list, picks: list, title: str):
    """Render a totals (over/under) section."""
    recommended = [p for p in picks if p.get("recommended")]
    others = [p for p in picks if not p.get("recommended")]

    lines.append(f"## {title}")
    lines.append("")

    if recommended:
        lines.append(f"### RECOMMENDED PLAYS ({len(recommended)})")
        lines.append("")
        lines.append("| Matchup | Pitchers | Line | Pick | Probability | Edge | Confidence | Notes |")
        lines.append("|---------|----------|------|------|-------------|------|------------|-------|")

        for p in recommended:
            home = p.get("home_team", "")
            away = p.get("away_team", "")
            home_sp = p.get("home_sp_name", "TBD")
            away_sp = p.get("away_sp_name", "TBD")
            line = p.get("line", "?")
            pick = p.get("pick", "?")
            prob = p.get("model_prob", 0)
            edge = p.get("edge_pct", "")
            conf = p.get("confidence", "")
            notes = p.get("notes", "")
            prob_str = f"{prob:.1%}" if isinstance(prob, float) else str(prob)

            lines.append(f"| {away} @ {home} | {away_sp} vs {home_sp} | {line} | {pick} | {prob_str} | {edge} | {conf} | {notes} |")

        lines.append("")
    else:
        lines.append("No recommended totals plays today.")
        lines.append("")

    if others:
        lines.append(f"**Other Games ({len(others)})** — below edge threshold:")
        lines.append("")
        lines.append("| Matchup | Line | Pick | Probability | Edge |")
        lines.append("|---------|------|------|-------------|------|")
        for p in others:
            home = p.get("home_team", "")
            away = p.get("away_team", "")
            line = p.get("line", "?")
            pick = p.get("pick", "?")
            prob = p.get("model_prob", 0)
            edge = p.get("edge_pct", "")
            prob_str = f"{prob:.1%}" if isinstance(prob, float) else str(prob)
            lines.append(f"| {away} @ {home} | {line} | {pick} | {prob_str} | {edge} |")
        lines.append("")

    lines.append("---")
    lines.append("")


if __name__ == "__main__":
    target = date.today()
    if len(sys.argv) > 1:
        try:
            target = datetime.strptime(sys.argv[1], "%Y-%m-%d").date()
        except ValueError:
            pass
    generate_picks_page(target)
