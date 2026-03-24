FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories
RUN mkdir -p data/raw data/processed data/output models

# Default command: run daily pipeline
CMD ["python", "run_daily.py"]
