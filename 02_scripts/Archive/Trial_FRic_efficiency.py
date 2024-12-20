#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimized script to compute functional richness (FRic) for large raster datasets
loaded from an S3 bucket.

Author: M. Hayden
Updated: December 2024
"""

import rasterio
import numpy as np
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from multiprocessing import Pool, cpu_count
import os
import boto3
import argparse
from rasterio.windows import Window
from tqdm import tqdm  # For progress tracking

# Set global parameters
Data_Dir = '/home/ec2-user/BioSCape_across_scales/01_data/02_processed'
Out_Dir = '/home/ec2-user/BioSCape_across_scales/03_output'
bucket_name = 'bioscape.gra'
comps = 3  # Number of PCA components
window_sizes = [60, 120, 240, 480, 960, 1200]  # Window sizes for FRic computation

# Initialize S3 client
s3 = boto3.client('s3')

# Function to download raster from S3
def download_from_s3(bucket, s3_key, local_path):
    try:
        s3.download_file(bucket, s3_key, local_path)
        print(f"Downloaded from S3: {s3_key}")
    except Exception as e:
        print(f"Failed to download from S3: {e}")
        raise

# Function to upload file to S3
def upload_to_s3(bucket, local_path, s3_key):
    try:
        s3.upload_file(local_path, bucket, s3_key)
        print(f"Uploaded to S3: {s3_key}")
    except Exception as e:
        print(f"Failed to upload to S3: {e}")

# Function to compute PCA on raster chunk
def process_chunk(chunk, nan_mask):
    # Impute missing values
    imputer = SimpleImputer(strategy='median')
    chunk_imputed = imputer.fit_transform(chunk)

    # Scale and standardize
    scaler = RobustScaler()
    chunk_scaled = scaler.fit_transform(chunk_imputed)

    # Perform PCA
    pca = PCA(n_components=comps)
    pca_transformed = pca.fit_transform(chunk_scaled)

    # Reapply NaN mask
    chunk_pca = pca_transformed.reshape((chunk.shape[0], -1, comps))
    chunk_pca[nan_mask] = np.nan

    return chunk_pca


# Main function to process the raster
def process_raster(input_raster, site_code):
    with rasterio.open(input_raster) as src:
        profile = src.profile
        width = src.width
        height = src.height
        count = src.count

        # Define block processing size
        block_size = 1024  # Process chunks of 1024x1024 pixels
        chunks = [(x, y, block_size, block_size)
                  for x in range(0, width, block_size)
                  for y in range(0, height, block_size)]

        # Prepare output raster for PCA components
        output_pca_path = os.path.join(Out_Dir, f"{site_code}_pca.tif")
        profile.update(dtype='float32', count=comps)
        with rasterio.open(output_pca_path, 'w', **profile) as dst_pca:
            for x, y, w, h in tqdm(chunks, desc="Processing chunks"):
                window = Window(x, y, w, h)
                data = src.read(window=window, out_shape=(count, h, w))
                data = data.transpose(1, 2, 0)  # Reorder to (rows, cols, bands)
                
                # Flatten and create mask
                data_flat = data.reshape(-1, count)
                nan_mask = np.isnan(data_flat)

                # Skip empty chunks
                if np.all(nan_mask):
                    continue

                # Process PCA for the chunk
                chunk_pca = process_chunk(data_flat, nan_mask)

                # Write PCA results to the output raster
                for i in range(comps):
                    dst_pca.write(chunk_pca[:, :, i], i + 1, window=window)

    print(f"PCA raster saved: {output_pca_path}")
    upload_to_s3(bucket_name, output_pca_path, f"{site_code}_pca.tif")

# Argument parser
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute PCA and FRic for large raster data.")
    parser.add_argument('--SITECODE', type=str, required=True, help='Site code (e.g., BART)')
    parser.add_argument('--INPUT_RASTER_KEY', type=str, required=True, help='S3 key for input raster file')
    args = parser.parse_args()

    # Download raster from S3
    local_raster_path = os.path.join(Data_Dir, "input_raster.tif")
    download_from_s3(bucket_name, args.INPUT_RASTER_KEY, local_raster_path)

    # Process raster
    process_raster(local_raster_path, args.SITECODE)
