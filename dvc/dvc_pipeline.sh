#!/bin/bash

# Exit on error
set -e

# Check if DVC is initialized (first run)
if [ ! -d ".dvc" ]; then
  echo "Initializing DVC..."
  dvc init
  
  echo "Setting up DVC remote storage..."
  dvc remote add -d myremote gcs://$GCS_BUCKET_NAME/dvc_store
  dvc remote modify myremote endpointurl https://storage.googleapis.com
  echo "DVC remote storage setup complete."
else
  echo "DVC is already initialized."
fi

# Pull the latest data (if needed)
echo "Pulling latest data from DVC remote..."
dvc pull

# Add new datasets to DVC tracking
echo "Adding rag dataset to DVC..."
dvc add /app/rag_dataset

echo "Adding cpsycoun dataset to DVC..."
dvc add /app/cpsycoun_dataset

# Push changes to remote storage
echo "Pushing datasets to remote storage..."
dvc push

echo "Finished DVC pipeline execution."
