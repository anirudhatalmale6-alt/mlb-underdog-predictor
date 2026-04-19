"""
Central configuration for the MLB Underdog Predictor.
All settings are loaded from environment variables with sensible defaults.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = DATA_DIR / "output"
MODELS_DIR = PROJECT_ROOT / "models"

for d in [RAW_DIR, PROCESSED_DIR, OUTPUT_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── API Keys ─────────────────────────────────────────────────────────────
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")

# ── Odds API ─────────────────────────────────────────────────────────────
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
ODDS_REGIONS = "us"
ODDS_MARKETS = "h2h"
ODDS_FORMAT = "american"

# ── MLB StatsAPI ─────────────────────────────────────────────────────────
MLB_API_BASE = "https://statsapi.mlb.com/api/v1"
MLB_SPORT_ID = 1  # MLB

# ── Underdog Criteria ────────────────────────────────────────────────────
MIN_UNDERDOG_ODDS = int(os.getenv("MIN_UNDERDOG_ODDS", "125"))
MAX_UNDERDOG_ODDS = int(os.getenv("MAX_UNDERDOG_ODDS", "250"))

# ── Model Settings ───────────────────────────────────────────────────────
EDGE_THRESHOLD = float(os.getenv("EDGE_THRESHOLD", "0.06"))
RETRAIN_DAILY = os.getenv("RETRAIN_DAILY", "false").lower() == "true"

# Training seasons for historical backtest
HISTORICAL_SEASONS = list(range(2019, 2025))  # 2019-2024
BACKTEST_TEST_SEASONS = [2022, 2023, 2024]

# XGBoost default hyperparameters (tuned via backtest)
XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 5,
    "learning_rate": 0.03,
    "n_estimators": 500,
    "min_child_weight": 3,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": 0,
}

# ── Feature Engineering ──────────────────────────────────────────────────
ROLLING_WINDOW_GAMES = 15       # Games for rolling stats
ROLLING_WINDOW_DAYS = 14        # Days for recent form
SP_RECENT_STARTS = 5            # Last N starts for SP recent form
EARLY_SEASON_BLEND_DAYS = 45    # Days into season before trusting current-year stats

# ── Logging ──────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# ── Output ───────────────────────────────────────────────────────────────
OUTPUT_FORMAT = os.getenv("OUTPUT_FORMAT", "both")  # csv, json, or both

# ── Team Abbreviation Mapping ────────────────────────────────────────────
# Maps various abbreviation styles to a canonical form
TEAM_ABBREVS = {
    "ARI": "ARI", "Arizona Diamondbacks": "ARI", "arizonadiamondbacks": "ARI",
    "ATL": "ATL", "Atlanta Braves": "ATL", "atlantabraves": "ATL",
    "BAL": "BAL", "Baltimore Orioles": "BAL", "baltimoreorioles": "BAL",
    "BOS": "BOS", "Boston Red Sox": "BOS", "bostonredsox": "BOS",
    "CHC": "CHC", "Chicago Cubs": "CHC", "chicagocubs": "CHC",
    "CWS": "CWS", "Chicago White Sox": "CWS", "chicagowhitesox": "CWS",
    "CIN": "CIN", "Cincinnati Reds": "CIN", "cincinnatireds": "CIN",
    "CLE": "CLE", "Cleveland Guardians": "CLE", "clevelandguardians": "CLE",
    "COL": "COL", "Colorado Rockies": "COL", "coloradorockies": "COL",
    "DET": "DET", "Detroit Tigers": "DET", "detroittigers": "DET",
    "HOU": "HOU", "Houston Astros": "HOU", "houstonastros": "HOU",
    "KC": "KC", "Kansas City Royals": "KC", "kansascityroyals": "KC",
    "LAA": "LAA", "Los Angeles Angels": "LAA", "losangelesangels": "LAA",
    "LAD": "LAD", "Los Angeles Dodgers": "LAD", "losangelesdodgers": "LAD",
    "MIA": "MIA", "Miami Marlins": "MIA", "miamimarlins": "MIA",
    "MIL": "MIL", "Milwaukee Brewers": "MIL", "milwaukeebrewers": "MIL",
    "MIN": "MIN", "Minnesota Twins": "MIN", "minnesotatwins": "MIN",
    "NYM": "NYM", "New York Mets": "NYM", "newyorkmets": "NYM",
    "NYY": "NYY", "New York Yankees": "NYY", "newyorkyankees": "NYY",
    "OAK": "OAK", "Oakland Athletics": "OAK", "oaklandathletics": "OAK",
    "PHI": "PHI", "Philadelphia Phillies": "PHI", "philadelphiaphillies": "PHI",
    "PIT": "PIT", "Pittsburgh Pirates": "PIT", "pittsburghpirates": "PIT",
    "SD": "SD", "San Diego Padres": "SD", "sandiegopadres": "SD",
    "SF": "SF", "San Francisco Giants": "SF", "sanfranciscogiants": "SF",
    "SEA": "SEA", "Seattle Mariners": "SEA", "seattlemariners": "SEA",
    "STL": "STL", "St. Louis Cardinals": "STL", "stlouiscardinals": "STL",
    "TB": "TB", "Tampa Bay Rays": "TB", "tampabayrays": "TB",
    "TEX": "TEX", "Texas Rangers": "TEX", "texasrangers": "TEX",
    "TOR": "TOR", "Toronto Blue Jays": "TOR", "torontobluejays": "TOR",
    "WSH": "WSH", "Washington Nationals": "WSH", "washingtonnationals": "WSH",
}

# MLB team ID to abbreviation (from statsapi)
TEAM_ID_TO_ABBREV = {
    109: "ARI", 144: "ATL", 110: "BAL", 111: "BOS", 112: "CHC",
    145: "CWS", 113: "CIN", 114: "CLE", 115: "COL", 116: "DET",
    117: "HOU", 118: "KC", 108: "LAA", 119: "LAD", 146: "MIA",
    158: "MIL", 142: "MIN", 121: "NYM", 147: "NYY", 133: "OAK",
    143: "PHI", 134: "PIT", 135: "SD", 137: "SF", 136: "SEA",
    138: "STL", 139: "TB", 140: "TEX", 141: "TOR", 120: "WSH",
}

ABBREV_TO_TEAM_ID = {v: k for k, v in TEAM_ID_TO_ABBREV.items()}

# ── Park Factors (runs, relative to 100 = neutral) ──────────────────────
# Source: ESPN/FanGraphs 2023-2024 average
PARK_FACTORS = {
    "ARI": 104, "ATL": 101, "BAL": 102, "BOS": 105, "CHC": 101,
    "CWS": 100, "CIN": 105, "CLE": 97, "COL": 114, "DET": 97,
    "HOU": 100, "KC": 100, "LAA": 98, "LAD": 97, "MIA": 94,
    "MIL": 101, "MIN": 102, "NYM": 97, "NYY": 104, "OAK": 96,
    "PHI": 102, "PIT": 97, "SD": 95, "SF": 96, "SEA": 95,
    "STL": 98, "TB": 97, "TEX": 103, "TOR": 100, "WSH": 100,
}

# FIP constant (league average, updated yearly; ~3.15 is typical)
FIP_CONSTANT = 3.15
