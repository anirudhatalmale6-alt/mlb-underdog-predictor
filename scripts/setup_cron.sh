#!/bin/bash
# Set up cron job for daily MLB predictions
# Runs at 10:00 AM ET (after lineups are posted)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Make run script executable
chmod +x "$SCRIPT_DIR/run_daily.sh"

# Add cron entry (10 AM ET = 14:00 UTC during EDT, 15:00 UTC during EST)
# Using 14:00 UTC as default — adjust if you're in a different timezone
CRON_LINE="0 14 * * * $SCRIPT_DIR/run_daily.sh"

# Check if cron entry already exists
(crontab -l 2>/dev/null | grep -q "run_daily.sh") && {
    echo "Cron entry already exists. Skipping."
    crontab -l | grep "run_daily"
    exit 0
}

# Add new cron entry
(crontab -l 2>/dev/null; echo "$CRON_LINE") | crontab -

echo "Cron job installed:"
echo "  $CRON_LINE"
echo ""
echo "To verify: crontab -l"
echo "To remove: crontab -e (and delete the line)"
echo ""
echo "Note: This runs at 10:00 AM ET (14:00 UTC)."
echo "Adjust the hour in the cron line if needed for your timezone."
