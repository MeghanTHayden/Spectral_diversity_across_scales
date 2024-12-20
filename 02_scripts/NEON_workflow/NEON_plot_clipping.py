#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script clips existing mosaics to the coordinates of the plot and creates a clip for the plot.

Author: M. Hayden
Updated: October 11, 2024

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
from rasterio.windows import from_bounds
from shapely.geometry import box
import pyproj
# Import supporting functions, functions for calculating FRic and FDiv
from S01_Functions import *
from S01_Moving_Window_FRIC import *
from S01_Moving_Window_FDiv import *

# Set directories
Data_Dir = '/home/ec2-user/BioSCape_across_scales/01_data/02_processed'
Out_Dir = '/home/ec2-user/BioSCape_across_scales/03_output'
bucket_name = 'bioscape.gra'
s3 = boto3.client('s3')

# Use arg parse for local variables
# Create the parser
parser = argparse.ArgumentParser(description="Input script for clipping mosaics to survey plots.")

# Add the arguments
parser.add_argument('--SITECODE', type=str, required=True, help='SITECODE (All caps)')
#parser.add_argument('--EPSG', type=str, required=True, help='EPSG (#####)')

# Parse the arguments
args = parser.parse_args()

# Assign the arguments to variables
SITECODE = args.SITECODE
#EPSG = args.EPSG

file_stem = SITECODE + '_flightlines/Mosaic_'

# Load coordinates file
# Define the S3 CSV path
csv_path_s3 = "s3://bioscape.gra/NEON_sr_summaries/NEON_species_summaries.csv"

# Load the CSV from S3 using pandas and s3fs
df = pd.read_csv(csv_path_s3)

# Identify plot IDs
# List shapefiles for a site in the S3 bucket in the matching directory
search_criteria = "Mosaic"
dirpath = SITECODE + "_flightlines/"
objects = s3.list_objects_v2(Bucket=bucket_name, Prefix=dirpath)['Contents']
# Filter objects based on the search criteria
shapefiles = [obj['Key'] for obj in objects if obj['Key'].endswith('.tif') and (search_criteria in obj['Key'])]
shapefile_names = set()
for i,shp in enumerate(shapefiles):
    match = re.search(r'_flightlines/Mosaic_(.*?).tif', shp)
    if match:
        shapefile_name = match.group(1)
        print(shapefile_name)
        shapefile_names.add(shapefile_name)
    else:
        print("Pattern not found in the URL.")
plots = list(shapefile_names)  # Convert set back to a list if needed
print(plots)

for i in plots:
    # Specify data
    clip_file = file_stem + str(i) + '.tif' # Define file name in S3
    print(clip_file)
    
    # Download plot mosaic
    s3.download_file(bucket_name, clip_file, Data_Dir + '/mosaic.tif')
    file = Data_Dir + '/mosaic.tif' # Define local file name
    print("Raster loaded")
    
    # Open as raster
    mosaic = rasterio.open(file)
    print(mosaic.crs)
    mosaicID = str(i)
    print(mosaicID)
    
    # Filter to get coordinates
    filtered_site = df[df['plotID'].apply(lambda x: x in mosaicID)]
    if filtered_site.empty:
        raise ValueError(f"Site {mosaicID} not found in the CSV.")
        continue
        
    # Get the latitude and longitude for the selected site
    lat = filtered_site.iloc[0]['latitude']
    lon = filtered_site.iloc[0]['longitude']
    coordinate = (lon, lat)

    # Transform the coordinate to the CRS of the raster (usually a projected CRS)
    project = pyproj.Transformer.from_crs("EPSG:4326", mosaic.crs, always_xy=True)  # Convert from WGS84 to the raster CRS
    x_proj, y_proj = project.transform(coordinate[0], coordinate[1])

    # Create a 20x20m bounding box (square) around the transformed coordinate
    size = 10  # 10 meters in each direction from the center
    square = box(x_proj - size, y_proj - size, x_proj + size, y_proj + size)

    # Clip the mosaic using the bounding box
    window = from_bounds(square.bounds[0], square.bounds[1], square.bounds[2], square.bounds[3], mosaic.transform)

    # Read the clipped data
    clipped_data = mosaic.read(window=window)

    # Update metadata for the clipped raster
    clipped_transform = mosaic.window_transform(window)
    clipped_meta = mosaic.meta.copy()
    clipped_meta.update({
        "driver": "GTiff",
        "height": clipped_data.shape[1],
        "width": clipped_data.shape[2],
        "transform": clipped_transform
    })

    # Save the clipped mosaic as a new file
    output_path = f"clipped_mosaic_{mosaicID}.tif"
    with rasterio.open(output_path, "w", **clipped_meta) as dest:
        dest.write(clipped_data)

    destination_s3_key_fric = "NEON_sr_summaries/Clip_" + mosaicID + ".tif"
    upload_to_s3(bucket_name, output_path, destination_s3_key_fric)

    os.remove(file)
    print("Mosaic clipped...next.")

