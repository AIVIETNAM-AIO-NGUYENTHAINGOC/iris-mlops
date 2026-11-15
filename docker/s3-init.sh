#!/bin/bash
# Create the bucket in local S3 and upload the trained model. Safe to run again.
set -euo pipefail

for _ in $(seq 1 30); do aws s3 ls > /dev/null 2>&1 && break; sleep 1; done # wait for Moto
aws s3api head-bucket --bucket iris-models 2> /dev/null || aws s3 mb s3://iris-models
aws s3 cp /artifacts/model.joblib s3://iris-models/iris/latest/model.joblib
