#!/bin/bash

set -e

# Define paths and configurations
export BASE_DIR=$(pwd)
export SECRETS_DIR=$(pwd)/../secrets/
export GCS_BUCKET_NAME="couch-gpt-store"
export GCP_PROJECT="couchgpt-453523"
export GCP_ZONE="us-central1"
export GOOGLE_APPLICATION_CREDENTIALS="/secrets/data-service-account.json"

# Pull the latest changes from the main branch
echo "Pulling latest changes from the main branch..."
git pull origin main --rebase  # ensures latest version

# Building Docker image
echo "Building image"
docker build -t data-version-cli -f Dockerfile .

# Running the container to perform DVC pipeline operations
echo "Running container"
docker run --rm --name data-version-cli -ti \
  --privileged \
  --cap-add SYS_ADMIN \
  --device /dev/fuse \
  -v "$BASE_DIR":/app \
  -v "$SECRETS_DIR":/secrets \
  -v ~/.gitconfig:/etc/gitconfig \
  -e GOOGLE_APPLICATION_CREDENTIALS=$GOOGLE_APPLICATION_CREDENTIALS \
  -e GCP_PROJECT=$GCP_PROJECT \
  -e GCP_ZONE=$GCP_ZONE \
  -e GCS_BUCKET_NAME=$GCS_BUCKET_NAME data-version-cli \
  bash -c "./dvc_pipeline.sh"

# Check if rag dataset was updated and stage
echo "Checking if rag dataset was updated..."
if [ -n "$(git status --porcelain /app/rag_dataset)" ]; then
  echo "Staging rag dataset changes..."
  git add /app/rag_dataset
  RAG_TAG="rag_dataset_v$(date +'%Y%m%d%H%M')"
  echo "Creating tag for rag dataset: $RAG_TAG"
  git tag -a "$RAG_TAG" -m "Tagging rag dataset update"
fi

# Check if cpsycoun dataset was updated and stage
echo "Checking if cpsycoun dataset was updated..."
if [ -n "$(git status --porcelain /app/cpsycoun_dataset)" ]; then
  echo "Staging cpsycoun dataset changes..."
  git add /app/cpsycoun_dataset
  CPSYCOUN_TAG="cpsycoun_dataset_v$(date +'%Y%m%d%H%M')"
  echo "Creating tag for cpsycoun dataset: $CPSYCOUN_TAG"
  git tag -a "$CPSYCOUN_TAG" -m "Tagging cpsycoun dataset update"
fi

# Commit and push changes (for either dataset or both)
echo "Committing changes..."
git commit -m "Dataset updates - $(date +'%Y-%m-%d %H:%M:%S')" || echo "No changes to commit"

# Push changes and respective tags
echo "Pushing changes and dataset tags..."
git push origin main
git push origin "$RAG_TAG" || echo "No rag dataset tag to push"
git push origin "$CPSYCOUN_TAG" || echo "No cpsycoun dataset tag to push"

echo "Pipeline execution complete!"
