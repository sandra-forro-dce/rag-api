#!/bin/bash
set -e

export BASE_DIR=$(pwd)
# export SECRETS_DIR=$(pwd)/../secrets/
export GCP_PROJECT="couchgpt-456015" 
export GOOGLE_APPLICATION_CREDENTIALS="/secrets/deployment.json"
export GCP_ZONE="us-central1-a"

# echo "Building image"
# docker build --no-cache -t agent -f Dockerfile .

# echo "Running container"
# docker run --rm  --name agent --network couchgpt -ti \
# -v "$BASE_DIR":/app \
# agent 

