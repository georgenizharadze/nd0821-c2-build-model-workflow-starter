#!/usr/bin/env python
"""
An example of a step using MLflow and Weights & Biases
"""
import argparse
import logging
import tempfile

import pandas as pd
import wandb


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    logger.info("Downloading artifact")
    artifact_local_path = run.use_artifact(args.input_artifact).file()

    logger.info("Reading data")
    df = pd.read_csv(artifact_local_path)

    logger.info("Cleaning data")
    df = df[df["price"].between(args.min_price, args.max_price)]
    idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
    df = df[idx]
    df['last_review'] = pd.to_datetime(df['last_review'])

    logger.info("Saving clean data locally")
    clean_data_path = 'clean_data.csv'
    df.to_csv(clean_data_path, index=False)

    logger.info("Uploading clean data artifact")
    artifact = wandb.Artifact(
        name=args.output_artifact,
        type=args.output_type,
        description=args.output_description
    )
    artifact.add_file(clean_data_path)
    run.log_artifact(artifact)
    logger.info("Finished uploading clean data artifact.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Clean raw data to align it with data expected in prod")


    parser.add_argument(
        "--input_artifact", 
        type=str,
        help="Input artifact",
        required=True
    )

    parser.add_argument(
        "--output_artifact", 
        type=str,
        help="Output artifact name",
        required=True
    )

    parser.add_argument(
        "--output_type", 
        type=str,
        help="Output artifact type",
        required=True
    )

    parser.add_argument(
        "--output_description", 
        type=str,
        help="Output artifact description",
        required=True
    )

    parser.add_argument(
        "--min_price", 
        type=float,
        help="Prices below this value considered outliers and removed",
        required=True
    )

    parser.add_argument(
        "--max_price", 
        type=float,
        help="Prices above this value considered outliers and removed",
        required=True
    )


    args = parser.parse_args()

    go(args)
