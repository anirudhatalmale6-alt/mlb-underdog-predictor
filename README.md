# MLB Underdog Predictor

Automated Python model that predicts when MLB underdogs (moneyline +130 to +250) are likely to win outright. Runs daily, pulls live data, and outputs recommended plays.

## Quick Start

### 1. Install

```bash
# Clone the repo
git clone https://github.com/anirudhatalmale6-alt/mlb-underdog-predictor.git
cd mlb-underdog-predictor

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Key

```bash
# Copy the example env file
cp .env.example .env

# Edit .env and add your Odds API key
# Get a free key at: https://the-odds-api.com
```

The Odds API free tier gives 500 requests/month — more than enough for daily use.
The MLB Stats API is free and requires no key.

### 3. Run Backtest (First Time)

Before using the model for daily picks, run the backtest to train the model:

```bash
python run_backtest.py
```

This will:
- Download historical game data from the MLB Stats API (takes 15-30 min first run)
- Build features for each historical game
- Train the XGBoost model using walk-forward validation
- Generate a backtest report with performance metrics
- Save the trained model to `models/`

### 4. Daily Predictions

```bash
# Today's picks
python run_daily.py

# Specific date
python run_daily.py 2024-08-15
```

Output is saved to `data/output/picks_YYYY-MM-DD.csv` and `.json`.

### 5. Automate with Cron

```bash
# Set up daily cron job (runs at 10 AM ET)
bash scripts/setup_cron.sh

# Or manually add to crontab:
# 0 14 * * * /path/to/mlb-predictor/scripts/run_daily.sh
```

### 6. Docker (Alternative)

```bash
# Build and run daily predictions
docker-compose up mlb-predictor

# Run backtest
docker-compose --profile backtest up backtest
```

## How It Works

### Data Sources
- **The Odds API** — Live moneyline odds from 40+ US bookmakers (requires free API key)
- **MLB Stats API** — Official MLB statistics: pitcher stats, team batting, standings, schedules (free, no key needed)

### Features (59 total)
The model uses these categories of features:

| Category | Examples |
|----------|---------|
| Starting Pitcher | ERA, WHIP, K/9, FIP, recent form (last 5 starts), quality start rate |
| Team Batting | OPS, ISO, K rate, BB rate, runs per game |
| Bullpen | ERA, WHIP, K/9 (team pitching minus SP contribution) |
| Momentum | Win %, Pythagorean win %, streak, last 10 record, run differential |
| Differentials | SP ERA gap, batting OPS gap, bullpen ERA gap, win % gap |
| Context | Park factor, home/away underdog, market implied probability |

### Model
- **XGBoost** classifier trained to predict underdog wins
- Walk-forward validation (train on prior seasons, test on next season)
- Outputs win probability for each qualifying underdog
- Recommends plays where model probability exceeds market-implied probability by the edge threshold

### Output Columns
| Column | Description |
|--------|-------------|
| underdog_team | Team abbreviation of the underdog |
| underdog_odds | American moneyline odds (e.g., +155) |
| market_implied_prob | What the betting market says the win probability is |
| model_win_prob | What the model predicts the win probability is |
| edge | model_win_prob minus market_implied_prob (positive = value) |
| confidence | HIGH (>10% edge), MEDIUM (>5%), LOW (>3%), NO PLAY |
| recommended | True if edge exceeds the configured threshold |
| notes | Brief analysis of why the model likes (or doesn't like) this play |

## Configuration

Edit `.env` to customize:

| Variable | Default | Description |
|----------|---------|-------------|
| ODDS_API_KEY | (required) | Your The Odds API key |
| EDGE_THRESHOLD | 0.03 | Minimum edge to recommend a play (0.03 = 3%) |
| MIN_UNDERDOG_ODDS | 130 | Minimum qualifying underdog odds |
| MAX_UNDERDOG_ODDS | 250 | Maximum qualifying underdog odds |
| OUTPUT_FORMAT | both | Output format: csv, json, or both |
| LOG_LEVEL | INFO | Logging verbosity |

## Project Structure

```
mlb-predictor/
├── run_daily.py          # Daily predictions entry point
├── run_backtest.py       # Backtest entry point
├── config/
│   └── settings.py       # All configuration
├── src/
│   ├── ingest/           # Data acquisition (MLB API, Odds API)
│   ├── features/         # Feature engineering (pitcher, batting, bullpen, momentum)
│   ├── model/            # XGBoost training, prediction, evaluation
│   ├── backtest/         # Walk-forward backtesting engine
│   ├── pipeline/         # Daily orchestration
│   └── utils/            # Odds math, date helpers, logging
├── data/
│   ├── output/           # Daily picks (CSV/JSON) + backtest reports
│   ├── processed/        # Cached feature matrices
│   └── raw/              # Raw API snapshots
├── models/               # Saved model artifacts
├── scripts/              # Cron and deployment scripts
├── Dockerfile
└── docker-compose.yml
```

## Important Notes

- **No guaranteed profits.** This is a statistical model, not a crystal ball. Use at your own risk.
- **Backtest ≠ future performance.** Historical results show what the model would have done, not what it will do.
- **Bankroll management matters.** Never bet more than you can afford to lose. The Kelly criterion output can help size bets, but use fractional Kelly (25-50% of full Kelly) for safety.
- **The model improves with more data.** Run the backtest periodically with updated seasons to retrain.
- **API costs:** The Odds API free tier (500 req/month) is sufficient. Paid plans start at ~$20/month if you need more.
