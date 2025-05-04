#!/bin/bash
set -e

# Run the application with error redirection
echo "=== Starting Agent ==="
# pipenv run python routing_agent_service.py 
pipenv run uvicorn agent:app --host 0.0.0.0 --port 8000
# echo "API is now running at http://localhost:9000/sft/inference"


