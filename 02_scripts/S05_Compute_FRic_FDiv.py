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
import empca
from sklearn.impute import SimpleImputer
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

    # Rescale data
    X /= 10000

    # Calculate mean and standard deviation while ignoring NaNs
    print("Proportion of NaN values:", np.isnan(X).mean())
    mean_vector = np.nanmean(X, axis=0)  # Mean of each band
    std_vector = np.nanstd(X, axis=0)  # Standard deviation of each band

    # Rescale the data: Center and scale each band
    X_scaled = (X - mean_vector) / std_vector

    # Replace NaN values with zeros for empca (empca handles them internally)
    X_scaled[np.isnan(X_scaled)] = 0

    # Apply PCA with empca
    pca_model = empca.PCA(X_scaled, n_components=comps)

    # Transform the data using the fitted PCA model
    pca_transformed = pca_model.transform(X_scaled)

    # Reshape PCA result back to raster dimensions
    pca_x = pca_transformed.reshape((dim1, dim2, comps))

    # Print PCA output shape
    print("PCA shape:", pca_x.shape)

    # Look at explained variance
    # Get singular values from the PCA model
    singular_values = pca_model.s  # Singular values from empca

    # Compute explained variance
    explained_variance = singular_values**2  # Variance for each principal component

    # Compute total variance
    total_variance = np.sum(explained_variance)

    # Compute explained variance ratio
    explained_variance_ratio = explained_variance / total_variance

    # Print explained variance and explained variance ratio
    print("Explained Variance:", explained_variance)
    print("Explained Variance Ratio:", explained_variance_ratio)
    
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
    destination_s3_key_fdiv = "/" + SITECODE + "_fdiv_veg_" + str(i) + ".csv"
    upload_to_s3(bucket_name, local_file_path_fdiv, destination_s3_key_fdiv)
    print("FDiv file uploaded to S3")

    # Remove files to clear storage
    os.remove(file)
    X = None
    X_no_nan = None
    pca_x = None
    veg_np = None
    
    print("Mosaic Complete - Next...")
