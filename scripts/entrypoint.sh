#!/bin/bash
set -e

echo "Prefect version:"
prefect version
# poetry run prefect version

until curl -s http://prefect-server:4200/api/health; do
    echo "Waiting"
    sleep 5
done

echo "Init prefect worker"
exec prefect worker start --pool my-docker-pool

# echo "Server  healthly"
# prefect work-pool create --type docker my-docker-pool || echo "Pool exist"

# exec prefect work-pool ls

# echo "Deploy flow"
# export PYTHONPATH="/app"
# poetry run python /app/prefect/flow.py
