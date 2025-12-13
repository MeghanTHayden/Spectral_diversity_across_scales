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
# Import supporting functions, functions for calculating FRic and FDiv
from S01_Functions import *
from S01_Moving_Window_FRIC import *
from S01_Moving_Window_FDiv import *

# Set directories
Data_Dir = '/home/ec2-user/BioSCape_across_scales/01_data/02_processed'
Out_Dir = '/home/ec2-user/BioSCape_across_scales/03_output'
bucket_name = 'bioscape.gra'
s3 = boto3.client('s3')

# Set global parameters #
# window_sizes = [10, 30, 60, 120]   # smaller list of window sizes to test
window_sizes = [60, 120, 240, 480, 960, 1200, 1500, 2000, 2200] # full list of window size for computations
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

# File to save PCA variance
VAR_OUT = os.path.join(Out_Dir, "PCA_variance_explained_NDVI_iter.csv")

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
#exclude = ["001", "004", "044", "009", "010"]
#plots = [p for p in plots if p not in exclude]
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

    bands, dim1, dim2 = shape[0], shape[1], shape[2]
  
    NIR_IDX = 96 - 1   # 95
    RED_IDX = 54 - 1   # 53

    nir = veg_np[NIR_IDX, :, :].astype('float32')
    red = veg_np[RED_IDX, :, :].astype('float32')

    ndvi = np.full(nir.shape, np.nan, dtype = 'float32')

    valid = (nir > 0) & (red > 0) & ((nir + red) != 0)

    ndvi[valid] = (nir[valid] - red[valid])/(nir[valid] + red[valid])

    # NDVI threshold
    ndvi_mask = (ndvi >= 0.40)
    #veg_np[:,~ndvi_mask] = np.nan
    print("Proportion of pixels passing NDVI >= 0.40:", np.nanmean(ndvi_mask))
    print("Proportion of pixels with valid NDVI inputs (nir>0 & red>0):", np.mean(valid))

    # Process for PCA
    # Flatten features into one dimension
    dim1, dim2, bands = shape[1], shape[2], shape[0]
    X = veg_np.reshape(bands, dim1 * dim2).T
    print("Shape of flattened array:", X.shape)

    # Set no data to nan
    X = X.astype('float32')
    #bad = np.all(X <= 0, axis = 1)
    #X[bad, :] = np.nan
    print("Proportion of NaN values:", np.isnan(X).mean())

    # Rescale data
    X /= 10000

    # Flatten NDVI mask to match X rows
    #ndvi_mask_flat = ndvi_mask.reshape(dim1 * dim2)

    # Only use vegetated pixels (NDVI >= 0.4) to fit imputer, scaler, and PCA
    #valid = ndvi_mask_flat & ~np.isnan(X).all(axis=1)
    #print("Number of valid (NDVI>=0.4) pixels:", valid.sum())

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

    # Save variance explained by each PC for this site & plot
    explained = pca.explained_variance_ratio_  # 1D array length = comps
    fieldnames = (
        ["site_code", "plot_id", "n_pc", "total_variance"] +
        [f"PC{k}" for k in range(1, comps + 1)]
    )
    row = {
        "site_code": SITECODE,
        "plot_id": str(i),
        "n_pc": comps,
        "total_variance": float(explained.sum())
    }
    for k, val in enumerate(explained, start=1):
        row[f"PC{k}"] = float(val)
    file_exists = os.path.isfile(VAR_OUT)
    with open(VAR_OUT, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    # PCA transform
    pca_x = pca.transform(X_transformed)
    pca_x = pca_x.reshape((dim1, dim2, comps))
    print("PCA shape:", pca_x.shape)
    
    # Calculate FRic on PCA across window sizes
    print("Calculating FRic")
    results_FR = {}
    local_file_path_fric = Out_Dir + "/" + SITECODE + "_fric_" + str(i) + ".csv"
    window_batches = [(a, pca_x, results_FR, local_file_path_fric) for a in np.array_split(window_sizes, cpu_count() - 1) if a.any()]
    volumes = process_map(
        window_calcs_old,
        window_batches,
        max_workers=cpu_count() - 1
    )
    destination_s3_key_fric = "/" + SITECODE + "_fric_pc3_ndvi_trial_" + str(i) + ".csv"
    upload_to_s3(bucket_name, local_file_path_fric, destination_s3_key_fric)
    print("FRic file uploaded to S3")
    
    # Calculate FDiv on PCA across window sizes
    print("Calculating FDiv")
    results_FD = {}
    local_file_path_fdiv = Out_Dir + "/" + SITECODE + "_fdiv_veg_" + str(i) + ".csv"
    window_batches = [(a, pca_x, results_FD, local_file_path_fdiv) for a in np.array_split(window_sizes, cpu_count() - 1) if a.any()]
    volumes = process_map(
        window_calcs_fdiv,
        window_batches,
        max_workers=cpu_count() - 1
    )
    # open file for writing
    destination_s3_key_fdiv = "/" + SITECODE + "_fdiv_pc3_ndvi_trial_" + str(i) + ".csv"
    upload_to_s3(bucket_name, local_file_path_fdiv, destination_s3_key_fdiv)
    print("FDiv file uploaded to S3")

    # Remove files to clear storage
    os.remove(file)
    X = None
    X_no_nan = None
    pca_x = None
    veg_np = None
    
    print("Mosaic Complete - Next...")
