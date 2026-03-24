#!/bin/bash
# Daily MLB Underdog Predictor runner script
# Add this to cron: 0 10 * * * /path/to/mlb-predictor/scripts/run_daily.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Activate virtual environment
source .venv/bin/activate

# Run daily pipeline
python run_daily.py >> "$PROJECT_DIR/data/output/daily_run.log" 2>&1

echo "[$(date)] Daily pipeline completed" >> "$PROJECT_DIR/data/output/daily_run.log"
