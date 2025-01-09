#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to compute functional richness and divergence for a set of moving windows across a raster.

Author: M. Hayden
Updated: May 20, 2024

User input:
1. Name of the NEON site (e.g., BART)
    
"""

# Load required libraries
import hytools as ht
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import numpy as np
import requests
import sklearn
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
import kneed
from kneed import KneeLocator
import scipy.spatial
from scipy.spatial import ConvexHull
import subprocess
from urllib.request import urlretrieve
import multiprocessing as mp
import os, glob
import csv
import rasterio
from osgeo import gdal
import rioxarray as rxr
import xarray as xr
import earthpy as et
import earthpy.spatial as es
import earthpy.plot as ep
import copy
import re
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ClientError
from shapely.geometry import box
import geopandas as gpd
import pandas as pd
from fiona.crs import from_epsg
import pycrs
import csv
from csv import writer
import argparse
from rasterio.transform import from_origin
# Import supporting functions, functions for calculating FRic and FDiv
from S01_Functions import *
from S01_Moving_Window_FRIC import *
from S01_Moving_Window_FDiv import *

def write_pca_to_raster(SITECODE, Data_Dir, pca_x):
    # Define the output raster file name
    local_pca_raster_path = os.path.join(Out_Dir, f"{SITECODE}_pca_{i}.tif")
    s3_pca_raster_key = f"{SITECODE}_flightlines/{SITECODE}_pca_{i}.tif"

    # Write PCA to raster
    with rasterio.open(
        local_pca_raster_path,
        "w",
        driver="GTiff",
        height=pca_x.shape[0],
        width=pca_x.shape[1],
        count=pca_x.shape[2],  # Number of components as raster bands
        dtype="float32",
        crs=raster.rio.crs,  # Use CRS from the original raster
        transform=raster.rio.transform(),  # Use transform from the original raster
    ) as dst:
        for band in range(pca_x.shape[2]):  # Loop through PCA components
            dst.write(pca_x[:, :, band], band + 1)

    print(f"PCA raster saved locally: {local_pca_raster_path}")

    # Upload the PCA raster to S3
    try:
        s3.upload_file(local_pca_raster_path, bucket_name, s3_pca_raster_key)
        print(f"PCA raster uploaded to S3: {s3_pca_raster_key}")
    except ClientError as e:
        print(f"Error uploading PCA raster to S3: {e}")

    # Clean up local PCA raster to save space
    os.remove(local_pca_raster_path)
    print(f"Local PCA raster removed: {local_pca_raster_path}")

# Set directories
Data_Dir = '/home/ec2-user/BioSCape_across_scales/01_data/02_processed'
Out_Dir = '/home/ec2-user/BioSCape_across_scales/03_output'
bucket_name = 'bioscape.gra'
s3 = boto3.client('s3')

# Set global parameters #
# window_sizes = [10, 30, 60, 120]   # smaller list of window sizes to test
#window_sizes = [60, 120, 240, 480, 960, 1200, 1500, 2000, 2200] # full list of window size for computations
window_sizes = [60, 90, 130, 195, 285, 420, 620, 920, 1350, 2000] # logarithmically spaced

comps = 3 # number of components for PCA

# Use arg parse for local variables
# Create the parser
parser = argparse.ArgumentParser(description="Input script for computing functional diversity metrics.")

# Add the arguments
parser.add_argument('--SITECODE', type=str, required=True, help='SITECODE (All caps)')

# Parse the arguments
args = parser.parse_args()

# Assign the arguments to variables
SITECODE = args.SITECODE

# Choose site and plots
file_stem = SITECODE + '_flightlines/Mosaic_' + SITECODE + '_'

# Identify plot IDs
# List shapefiles for a site in the S3 bucket in the matching directory
search_criteria = "Mosaic_"
dirpath = SITECODE + "_flightlines/"
objects = s3.list_objects_v2(Bucket=bucket_name, Prefix=dirpath)['Contents']
# Filter objects based on the search criteria
mosaics = [obj['Key'] for obj in objects if obj['Key'].endswith('.tif') and (search_criteria in obj['Key'])]
mosaic_names = set()
for i,tif in enumerate(mosaics):
    match = re.search(r'(.*?)_flightlines/Mosaic_(.*?)_(.*?).tif', tif)
    if match:
        mosaic_name = match.group(3)
        print(mosaic_name)
        mosaic_names.add(mosaic_name)
    else:
        print("Pattern not found in the URL.")
plots = list(mosaic_names)  # Convert set back to a list if needed
print(plots)

# Loop through plots to calculate FRic and FDiv
for i in plots:
    # Load data
    clip_file = file_stem + str(i) + '.tif' # Define file name in S3
    print(clip_file)
    # Download plot mosaic
    s3.download_file(bucket_name, clip_file, Data_Dir + '/mosaic.tif')
    file = Data_Dir + '/mosaic.tif' # Define local file name
    print("Raster loaded")
    # Open as raster
    raster = rxr.open_rasterio(file, masked=True)
    print(raster)
    # Convert data array to numpy array
    veg_np = raster.to_numpy()
    shape = veg_np.shape
    print("Raster shape:", shape)
    
    # Process for PCA
    # Flatten features into one dimension
    dim1, dim2, bands = shape[1], shape[2], shape[0]
    X = veg_np.reshape(bands, dim1 * dim2).T
    print("Shape of flattened array:", X.shape)

    # Set no data to nan
    X = X.astype('float32')
    X[np.isnan(X)] = np.nan
    X[X <= 0] = np.nan  # Adjust threshold if needed
    # Save nan mask
    nan_mask = np.isnan(X)
    print("Proportion of NaN values:", np.isnan(X).mean())

    # Rescale data
    X /= 10000

    # Impute missing values
    imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    X_transformed = imputer.fit_transform(X)

    # Scale & standardize array
    scaler = RobustScaler()
    X_transformed = scaler.fit_transform(X_transformed)

    # Perform initial PCA fit
    print("Fitting PCA")
    pca = PCA(n_components=comps)
    pca.fit(X_transformed)
    print("Explained variance ratio:", pca.explained_variance_ratio_)

    # PCA transform
    pca_x = pca.transform(X_transformed)
    pca_x = pca_x.reshape((dim1, dim2, comps))
    print("PCA shape:", pca_x.shape)

    # Reapply NaN mask
    nan_mask_reshaped = nan_mask[:, 0].reshape(dim1, dim2)  # Ensure mask matches raster dimensions
    for i in range(comps):  # Apply mask to each PCA component
        pca_x[:, :, i][nan_mask_reshaped] = np.nan

    write_pca_to_raster(SITECODE, Data_Dir, pca_x)

    # Randomize pixels to remove spatial organization
    print("Randomizing pixels for null distribution...")
    pca_x_flat = pca_x.reshape(-1, comps)  # Flatten PCA array to 2D (n_pixels, n_components)
    np.random.shuffle(pca_x_flat)          # Shuffle rows randomly
    pca_x_random = pca_x_flat.reshape(dim1, dim2, comps)  # Reshape back to original dimensions
    print("Randomization complete. Shape:", pca_x_random.shape)

    # Calculate FRic on PCA across window sizes
    print("Calculating FRic")
    results_FR = {}
    local_file_path_fric = Out_Dir + "/" + SITECODE + "_fric_" + str(i) + ".csv"
    window_batches = [(a, pca_x, results_FR, local_file_path_fric) for a in np.array_split(window_sizes, cpu_count() - 1) if a.any()]
    volumes = process_map(
        window_calcs,
        window_batches,
        max_workers=cpu_count() - 1
    )
    destination_s3_key_fric = "/" + SITECODE + "_fric_veg_" + str(i) + ".csv"
    upload_to_s3(bucket_name, local_file_path_fric, destination_s3_key_fric)
    print("FRic file uploaded to S3")

    # Calculate FRic with randomly distributed pixels from PCA across window sizes
    print("Calculating FRic")
    results_FR = {}
    local_file_path_fric = Out_Dir + "/" + SITECODE + "_fric_null_" + str(i) + ".csv"
    window_batches = [(a, pca_x_random, results_FR, local_file_path_fric) for a in np.array_split(window_sizes, cpu_count() - 1) if a.any()]
    volumes = process_map(
        window_calcs,
        window_batches,
        max_workers=cpu_count() - 1
    )
    destination_s3_key_fric = "/" + SITECODE + "_fric_veg_null_" + str(i) + ".csv"
    upload_to_s3(bucket_name, local_file_path_fric, destination_s3_key_fric)
    print("FRic file uploaded to S3")
    
    # Calculate FDiv on PCA across window sizes
    print("Calculating FDiv")
    results_FD = {}
    local_file_path_fdiv = Out_Dir + "/" + SITECODE + "_fdiv_veg_null_" + str(i) + ".csv"
    window_batches = [(a, pca_x_random, results_FD, local_file_path_fdiv) for a in np.array_split(window_sizes, cpu_count() - 1) if a.any()]
    volumes = process_map(
        window_calcs_fdiv,
        window_batches,
        max_workers=cpu_count() - 1
    )
    # open file for writing
    destination_s3_key_fdiv = "/" + SITECODE + "_fdiv_veg_null_" + str(i) + ".csv"
    upload_to_s3(bucket_name, local_file_path_fdiv, destination_s3_key_fdiv)
    print("FDiv file uploaded to S3")

    # Remove files to clear storage
    os.remove(file)
    X = None
    X_no_nan = None
    pca_x = None
    pca_x_random = None
    veg_np = None
    
    print("Mosaic Complete - Next...")
