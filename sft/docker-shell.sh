#!/bin/bash
set -e

export BASE_DIR=$(pwd)
export SECRETS_DIR=$(pwd)/../secrets/
export GCP_PROJECT="couchgpt-456015" 
export GOOGLE_APPLICATION_CREDENTIALS="/secrets/deployment.json"
export GCP_ZONE="us-central1-a"


# Echo system information
echo "=== System Information ==="
echo "Host CUDA Version:"
nvidia-smi | grep "CUDA Version"
echo "Available GPU Memory:"
nvidia-smi --query-gpu=memory.free,memory.total --format=csv

# echo "Building image"
# docker build --no-cache -t couchgpt -f Dockerfile .

# echo "Running container"
# docker run --rm --gpus all --name couchgpt -ti \
# -v "$BASE_DIR":/app \
# -v "$SECRETS_DIR":/secrets \
# -e GOOGLE_APPLICATION_CREDENTIALS=$GOOGLE_APPLICATION_CREDENTIALS \
# -e GCP_PROJECT=$GCP_PROJECT \
# -e GCP_ZONE=$GCP_ZONE \
# --shm-size=16g \
# --ipc=host \
# --ulimit stack=67108864 \
# --ulimit memlock=-1 \
# couchgpt #2>&1 | tee docker_output.log

# Check exit code
# EXIT_CODE=${PIPESTATUS[0]}
# if [ $EXIT_CODE -ne 0 ]; then
#   echo "=== Container exited with code $EXIT_CODE ==="
#   echo "Check docker_output.log for details"
# fi

# exit $EXIT_CODE