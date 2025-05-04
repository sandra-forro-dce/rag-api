#!/bin/bash

# exit immediately if a command exits with a non-zero status
set -e

# Set vairables
export BASE_DIR=$(pwd)
# export PERSISTENT_DIR=$(pwd)/../persistent-folder/
export SECRETS_DIR=$(pwd)/../secrets/
export GCP_PROJECT="couchgpt-456015" 
export GOOGLE_APPLICATION_CREDENTIALS="/secrets/deployment.json"
export IMAGE_NAME="couchgpt-rag-cli"


# # Create the network if we don't have it yet
# docker network inspect couchgpt >/dev/null 2>&1 || docker network create couchgpt

# # Build the image based on the Dockerfile
# docker build -t $IMAGE_NAME -f Dockerfile .

# # Run All Containers
# docker-compose run --rm --service-ports $IMAGE_NAME
