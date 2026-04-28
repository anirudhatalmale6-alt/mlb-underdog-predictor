#!/usr/bin/env python3
"""
Verify player props against confirmed MLB lineups.
Runs ~1 PM ET after lineups are posted.
Removes batter props for players not in the starting lineup,
and pitcher props for pitchers no longer listed as probable starters.
"""

import sys
import os
import json
import requests
import time
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import OUTPUT_DIR, MLB_API_BASE

PROJECT_ROOT = Path(__file__).resolve().parent


def get_game_lineups(game_pk: int) -> dict:
    """Fetch confirmed lineup for a game. Returns dict with player IDs in lineup."""
    try:
        url = f"{MLB_API_BASE}/game/{game_pk}/boxscore"
        r = requests.get(url, timeout=15)
        if r.status_code != 200:
            return {}
        data = r.json()

        lineup = {"home_batters": set(), "away_batters": set(),
                  "home_pitcher": None, "away_pitcher": None}

        for side in ["home", "away"]:
            team_data = data.get("teams", {}).get(side, {})

            batting_order = team_data.get("battingOrder", [])
            for player_id in batting_order:
                lineup[f"{side}_batters"].add(player_id)

            pitchers = team_data.get("pitchers", [])
            if pitchers:
                lineup[f"{side}_pitcher"] = pitchers[0]

        time.sleep(0.3)
        return lineup
    except Exception:
        return {}


def get_probable_pitchers(game_date: date) -> dict:
    """Fetch today's probable pitchers. Returns {game_pk: {home_sp_id, away_sp_id, home_sp_name, away_sp_name}}."""
    try:
        url = f"{MLB_API_BASE}/schedule"
        r = requests.get(url, params={
            "sportId": 1,
            "date": game_date.strftime("%Y-%m-%d"),
            "hydrate": "probablePitcher",
        }, timeout=15)
        if r.status_code != 200:
            return {}

        result = {}
        for date_entry in r.json().get("dates", []):
            for game in date_entry.get("games", []):
                gpk = game.get("gamePk")
                home = game.get("teams", {}).get("home", {})
                away = game.get("teams", {}).get("away", {})
                result[gpk] = {
                    "home_sp_id": home.get("probablePitcher", {}).get("id"),
                    "home_sp_name": home.get("probablePitcher", {}).get("fullName", "TBD"),
                    "away_sp_id": away.get("probablePitcher", {}).get("id"),
                    "away_sp_name": away.get("probablePitcher", {}).get("fullName", "TBD"),
                }
        return result
    except Exception:
        return {}


def get_schedule_game_pks(game_date: date) -> dict:
    """Get game PKs mapped by team matchup for cross-referencing."""
    try:
        from config.settings import TEAM_ID_TO_ABBREV
        url = f"{MLB_API_BASE}/schedule"
        r = requests.get(url, params={
            "sportId": 1,
            "date": game_date.strftime("%Y-%m-%d"),
        }, timeout=15)
        if r.status_code != 200:
            return {}

        result = {}
        for date_entry in r.json().get("dates", []):
            for game in date_entry.get("games", []):
                gpk = game.get("gamePk")
                home_id = game.get("teams", {}).get("home", {}).get("team", {}).get("id")
                away_id = game.get("teams", {}).get("away", {}).get("team", {}).get("id")
                home = TEAM_ID_TO_ABBREV.get(home_id, "")
                away = TEAM_ID_TO_ABBREV.get(away_id, "")
                if home and away:
                    result[(home, away)] = gpk
        return result
    except Exception:
        return {}


def search_player_id(name: str) -> int:
    """Search MLB API for player ID by name."""
    try:
        url = f"{MLB_API_BASE}/people/search"
        r = requests.get(url, params={"names": name, "sportId": 1}, timeout=10)
        if r.status_code != 200:
            return None
        people = r.json().get("people", [])
        if people:
            return people[0]["id"]
    except Exception:
        pass
    return None


def verify_and_filter(target_date: date = None):
    if target_date is None:
        target_date = date.today()

    date_str = target_date.strftime("%Y-%m-%d")
    json_path = OUTPUT_DIR / f"picks_{date_str}.json"

    if not json_path.exists():
        print(f"No picks file found for {date_str}")
        return

    with open(json_path) as f:
        picks = json.load(f)

    if not picks:
        print("No picks to verify")
        return

    prop_types = {"PITCHER K", "PITCHER OUTS", "BATTER HITS"}
    prop_picks = [p for p in picks if p.get("bet_type") in prop_types]
    other_picks = [p for p in picks if p.get("bet_type") not in prop_types]

    if not prop_picks:
        print("No player prop picks to verify")
        return

    print(f"Verifying {len(prop_picks)} player prop picks against confirmed lineups...")

    game_pks = get_schedule_game_pks(target_date)
    probable = get_probable_pitchers(target_date)

    games_needing_lineup = set()
    for p in prop_picks:
        home = p.get("home_team", "")
        away = p.get("away_team", "")
        gpk = game_pks.get((home, away))
        if gpk:
            games_needing_lineup.add(gpk)

    lineups = {}
    for gpk in games_needing_lineup:
        lu = get_game_lineups(gpk)
        if lu:
            lineups[gpk] = lu

    verified = []
    removed = []

    for p in prop_picks:
        home = p.get("home_team", "")
        away = p.get("away_team", "")
        gpk = game_pks.get((home, away))
        bet_type = p.get("bet_type", "")
        player_name = p.get("player_name", "")

        if not gpk:
            verified.append(p)
            continue

        if bet_type in ("PITCHER K", "PITCHER OUTS"):
            sp_data = probable.get(gpk, {})
            pitcher_confirmed = False

            for side in ["home", "away"]:
                sp_name = sp_data.get(f"{side}_sp_name", "")
                if sp_name and sp_name != "TBD" and _name_match(player_name, sp_name):
                    pitcher_confirmed = True
                    break

            if pitcher_confirmed:
                if "notes" in p and p["notes"]:
                    p["notes"] = p["notes"] + "; Starter confirmed"
                else:
                    p["notes"] = "Starter confirmed"
                verified.append(p)
            else:
                if sp_data:
                    removed.append(p)
                    print(f"  REMOVED: {player_name} ({bet_type}) - not confirmed as starter")
                else:
                    verified.append(p)

        elif bet_type == "BATTER HITS":
            lu = lineups.get(gpk, {})
            home_batters = lu.get("home_batters", set())
            away_batters = lu.get("away_batters", set())
            all_batters = home_batters | away_batters

            if not all_batters:
                verified.append(p)
                continue

            player_id = p.get("batter_id")
            if not player_id:
                player_id = search_player_id(player_name)

            if player_id and player_id in all_batters:
                if "notes" in p and p["notes"]:
                    p["notes"] = p["notes"] + "; In starting lineup"
                else:
                    p["notes"] = "In starting lineup"
                verified.append(p)
            elif player_id and all_batters:
                removed.append(p)
                print(f"  REMOVED: {player_name} ({bet_type}) - not in starting lineup")
            else:
                verified.append(p)
        else:
            verified.append(p)

    final_picks = other_picks + verified

    with open(json_path, "w") as f:
        json.dump(final_picks, f, indent=2)

    also_csv = OUTPUT_DIR / f"picks_{date_str}.csv"
    if final_picks:
        import pandas as pd
        pd.DataFrame(final_picks).to_csv(also_csv, index=False)

    print(f"\nLineup verification complete:")
    print(f"  Verified: {len(verified)} prop picks kept")
    print(f"  Removed:  {len(removed)} prop picks (player not in lineup)")
    print(f"  Total picks remaining: {len(final_picks)}")

    if removed:
        print("\nRemoved picks:")
        for p in removed:
            print(f"  - {p.get('player_name')} | {p.get('bet_type')} | {p.get('pick')}")


def _name_match(name1: str, name2: str) -> bool:
    if not name1 or not name2:
        return False
    n1 = name1.lower().strip()
    n2 = name2.lower().strip()
    if n1 == n2:
        return True
    last1 = n1.split()[-1] if n1 else ""
    last2 = n2.split()[-1] if n2 else ""
    return last1 == last2 and len(last1) > 2


if __name__ == "__main__":
    target = date.today()
    if len(sys.argv) > 1:
        try:
            target = datetime.strptime(sys.argv[1], "%Y-%m-%d").date()
        except ValueError:
            pass
    verify_and_filter(target)
