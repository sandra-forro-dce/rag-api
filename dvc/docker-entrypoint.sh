#!/bin/bash

echo "Container is running!!!"


gcloud auth activate-service-account --key-file=$GOOGLE_APPLICATION_CREDENTIALS
mkdir -p /mnt/gcs_bucket
gcsfuse --key-file=$GOOGLE_APPLICATION_CREDENTIALS $GCS_BUCKET_NAME /mnt/gcs_data
echo 'GCS bucket mounted at /mnt/gcs_data'
mkdir -p /app/rag_dataset
mount --bind /mnt/gcs_data/rag /app/rag_dataset
mkdir -p /app/cpsycoun_dataset
mount --bind /mnt/gcs_data/cpsycoun /app/cpsycoun_dataset

pipenv shell