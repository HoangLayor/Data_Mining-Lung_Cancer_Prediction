FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends gcc libpq-dev curl && rm -rf /var/lib/apt/lists/*

COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything to ensure src and config paths are available
COPY . .

# Ensure data and models dir exist for config paths
RUN mkdir -p data/raw data/processed models config

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
