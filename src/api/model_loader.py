"""Load the model from a local file (approach A) or S3 (approach B), chosen by MODEL_SOURCE."""

import io
import os
from urllib.parse import urlparse

import boto3
import joblib

LOCAL_PATH = "artifacts/model.joblib"


def load_model():
    source = os.getenv("MODEL_SOURCE", "local")
    if source == "local":
        return joblib.load(os.getenv("MODEL_PATH", LOCAL_PATH))
    if source == "s3":
        uri = urlparse(os.environ["MODEL_URI"])  # s3://<bucket>/<key>
        obj = boto3.client("s3").get_object(Bucket=uri.netloc, Key=uri.path.lstrip("/"))
        # joblib.load can execute code in the file: only load from our bucket, which only CI can write.
        return joblib.load(io.BytesIO(obj["Body"].read()))
    raise ValueError(f"MODEL_SOURCE must be local or s3, got {source!r}")
