FROM prefecthq/prefect:3-python3.12

RUN apt-get update && apt-get install -y gcc git libpq-dev curl && rm -rf /var/lib/apt/lists/*

# WORKDIR /app

# COPY pyproject.toml poetry.lock ./
# RUN pip install poetry && poetry install --no-root

# COPY src/ .
COPY scripts/entrypoint.sh /entrypoint.sh

RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]