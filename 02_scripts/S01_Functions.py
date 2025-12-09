#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script containing processing functions to:
    - plot an RGB map of multi-band raster.
    - find and load NEON files
    - subsample pixels and scale, center and fit PCA to subsample
    - parallelize various functions
    - pull from and upload to AWS S3 Bucket


Author: M. Hayden
Last Updated: 1/23/24
"""

## Load necessary packages ##
#import hytools as ht
import matplotlib.pyplot as plt
import numpy as np
import requests
import sklearn
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
#import kneed
#from kneed import KneeLocator
from scipy.spatial import ConvexHull
import subprocess
from urllib.request import urlretrieve
import parmap
import os
import tqdm 
from progress.bar import Bar
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ClientError
from tqdm.contrib.concurrent import process_map
from multiprocessing import Pool, cpu_count
from S01_Moving_Window_FRIC import * # add scripts folder to python path manager
import tqdm 
from progress.bar import Bar
from tqdm.contrib.concurrent import process_map
from multiprocessing import Pool, cpu_count
from osgeo import gdal, osr
from matplotlib.pyplot import subplots, show
import h5py
import sys
import rasterio
import boto3
import re
import pyproj
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ClientError

# Add global parameters for use of s3
Data_Dir = '/home/ec2-user/BioSCape_across_scales/01_data/01_rawdata'
Out_Dir = '/home/ec2-user/BioSCape_across_scales/01_data/02_processed'
bucket_name = 'bioscape.gra'
s3 = boto3.client('s3')

## 1. Interacting with NEON Database ####

# Find NEON files with specifications
def find_neon_files(SITECODE, PRODUCTCODE, 
                    YEAR):
    """Function to create API URLs for NEON images of interest.
    
    Parameters:
    -----------
    site: 
        String...
    product: 
        String...
    year:
        String...
        
    Returns:
    -----------
    file_paths: list of file paths for desired NEON images.
        URL for API where image is located.
    
    
    """
    # Server's URL will remain the same
    SERVER = 'http://data.neonscience.org/api/v0/'
    # Build url with user inputs
    url = SERVER+'data/'+PRODUCTCODE+'/'+SITECODE+'/'+YEAR
    # Request the url
    data_request = requests.get(url)
    # Convert the request to Python JSON object
    data_json = data_request.json()
    # Create list of file paths of interest for given site, product, year
    file_paths = []
    for file in data_json['data']['files'][:20]:
      if 'reflectance.h5' in file['name']:
            file_paths.insert(1, file['url'])
            print(file['url'])
    return file_paths

# Retrieve NEON files from API
def retrieve_neon_files(file_paths, data_directory):
    """Function to download files from list of file paths.
    
    Parameters:
    -----------
    
    file_paths: list of strings
    
    data_directory: string of location on OS to save data

        
    Returns:
    -----------
    files: list of locations of downloaded files on OS
    
    """
    files = []
    for file_path in file_paths:
        # for each file path, join path name with directory where it should be stored
        base_name = os.path.basename(file_path)
        save_path = os.path.join(data_directory, base_name)
        # retrieve files
        loc, message = urlretrieve(file_path, save_path)
        # add location of file to list
        files.append(loc)
    return files

## 2. Interacting with AWS S3 ####

# Download shapefile from S3 Bucket
def download_shapefile(bucket, prefix, output_dir):
    """Function to download all necessary files to constitute a shapefile from AWS S3 Bucket.
    
    Parameters:
    -----------
    
    bucket: string of bucket name
    
    prefix: string identifying which set of files should be downloaded

    output_dir: string of location in which to store files locally
   
    Returns:
    -----------
    files: list of locations of downloaded files on OS
    
    """
    # List all files with the given prefix
    files = s3.list_objects(Bucket=bucket, Prefix=prefix)['Contents']

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    downloaded_files = []
    
    # Download all files
    for file in files:
        key = file['Key']
        local_path = os.path.join(output_dir, os.path.basename(key))
        s3.download_file(bucket, key, local_path)
        downloaded_files.append(local_path)

    return downloaded_files

# Upload file to S3 Bucket
def upload_to_s3(bucket_name, file_path, s3_key):
    """
    Upload a file from an EC2 instance to Amazon S3.

    bucket_name: Name of the S3 bucket
    file_path: Local path to the file on the EC2 instance
    s3_key: Destination key in the S3 bucket (e.g., folder/file_name.ext)
    """
    # Initialize the S3 client
    s3 = boto3.client('s3')

    try:
    # Upload the file
        s3.upload_file(file_path, bucket_name, s3_key)
        print(f'Successfully uploaded {file_path} to {bucket_name}/{s3_key}')
    except Exception as e:
        print(f"Error uploading file: {e}")


## 3. Processing Images ####

# Convert array to raster (single band)
def array2raster(newRaster, reflBandArray, reflArray_metadata, Out_Dir, epsg):
    NP2GDAL_CONVERSION = {
        "uint8": 1,
        "int8": 1,
        "uint16": 2,
        "int16": 3,
        "uint32": 4,
        "int32": 5,
        "float32": 6,
        "float64": 7,
        "complex64": 10,
        "complex128": 11,
    }
    pwd = os.getcwd()
    os.chdir(Out_Dir)
    cols = reflBandArray.shape[1]
    rows = reflBandArray.shape[0]
    bands = 1
    pixelWidth = float(refl_md['res']['pixelWidth'])
    pixelHeight = -float(refl_md['res']['pixelHeight'])
    originX = refl_md['ext_dict']['xMin']
    originY = refl_md['ext_dict']['yMax']
    driver = gdal.GetDriverByName('GTiff')
    gdaltype = NP2GDAL_CONVERSION[reflBandArray.dtype.name]
    outRaster = driver.Create(newRaster, cols, rows, bands, gdaltype)
    outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
    # outband = outRaster.GetRasterBand(1)
    # outband.WriteArray(reflBandArray[:,:,x])
    #for band in range(bands):
    #    outRaster.GetRasterBand(band + 1).WriteArray(array[:, :, band])
    outRaster.WriteArray(reflBandArray[:, :])
    outRasterSRS = osr.SpatialReference()
    #outRasterSRS.ImportFromEPSG(reflArray_metadata['epsg'])
    #outRasterSRS.ExportToWkt()
    outRasterSRS.ImportFromEPSG(epsg)
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    outRaster.FlushCache()
    os.chdir(pwd)

# Convert array to raster (multi-band)
def array2rastermb(newRaster, reflBandArray, reflArray_metadata, Out_Dir, epsg, bands):
    NP2GDAL_CONVERSION = {
        "uint8": 1,
        "int8": 1,
        "uint16": 2,
        "int16": 3,
        "uint32": 4,
        "int32": 5,
        "float32": 6,
        "float64": 7,
        "complex64": 10,
        "complex128": 11,
    }
    pwd = os.getcwd()
    print(pwd)
    os.chdir(Out_Dir)
    cols = reflBandArray.shape[1]
    rows = reflBandArray.shape[0]
    print(cols,rows)
    pixelWidth = float(reflArray_metadata['res']['pixelWidth'])
    pixelHeight = -float(reflArray_metadata['res']['pixelHeight'])
    originX = reflArray_metadata['ext_dict']['xMin']
    originY = reflArray_metadata['ext_dict']['yMax']
    driver = gdal.GetDriverByName('GTiff')
    gdaltype = NP2GDAL_CONVERSION[reflBandArray.dtype.name]
    outRaster = driver.Create(newRaster, cols, rows, bands, gdaltype)
    print(outRaster)
    outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
    #outband = outRaster.GetRasterBand(1)
    #outband.WriteArray(reflBandArray[:,:,x])
    for band in range(bands):
        print(band)
        outRaster.GetRasterBand(band + 1).WriteArray(reflBandArray[:, :, band])
    #outRaster.WriteArray(reflBandArray[:, :, :])
    outRasterSRS = osr.SpatialReference()
    #outRasterSRS.ImportFromEPSG(reflArray_metadata['epsg'])
    #outRasterSRS.ExportToWkt()
    outRasterSRS.ImportFromEPSG(epsg)
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    outRaster.FlushCache()
    os.chdir(pwd)

# Clip raster to set bounds
def clip_raster(src, minx, miny, maxx, maxy):
    geom = box(minx, miny, maxx, maxy)
    out_image, out_transform = rasterio.mask(src, [geom], crop=True)
    out_meta = src.meta.copy()
    out_meta.update({"driver": "GTiff", 
        "height": out_image.shape[1],
        "width": out_image.shape[2],
        "transform": out_transform})
    return out_image, out_meta

# Scale and transform array with Principal Component Analysis 
def scale_transform(X, comps):
    """Function to center, scale and fit PCA transform.
    
    Parameters:
    -----------
    X:
        Numpy array. Subsampled pixels of HyTools object.
    
    comps:
        Integer. Number of principle components to fit.
        
    Returns:
    -----------
    x_mean:
        Numpy array. Mean reflectance across bands in subsampled pixels.
    x_std:
        Numpy array. Std reflectance across bands in subsampled pixels.
    pca:
        PCA transformation.
    
    """
    # Center, scale and fit PCA transform - scales based on mean reflectance at each band
    x_mean = X.mean(axis=0)[np.newaxis,:]
    X = X.astype('float32') # necessary to manually convert to float for next function to work
    X -=x_mean
    x_std = X.std(axis=0,ddof=1)[np.newaxis,:]
    X /=x_std
    X = X[~np.isnan(X.sum(axis=1)) & ~np.isinf(X.sum(axis=1)),:]
    # Perform initial PCA fit
    pca = PCA(n_components=comps) # set max number of components
    pca.fit(X)
    return x_mean, x_std, pca

# Full steps surrounding PCA transformation in FRic computation script
def pca_steps(raster, comps):
    """Function to center, scale and fit PCA transform.
    
    Parameters:
    -----------
    raster:
        Data Array. 
    
    comps:
        Integer. Number of principle components to fit.
        
    Returns:
    -----------
    pca_x: PCA transformed raster.
    
    """

    # Convert data array to numpy array
    veg_np = raster.to_numpy()
    shape = veg_np.shape
    print(shape)
    # Flatten features into one dimesnion
    dim1 = shape[1]
    dim2 = shape[2]
    bands = shape[0]
    X = veg_np.reshape(bands,dim1*dim2).T
    print(X.shape)
    X = X.astype('float32')
    X[np.isnan(X)] = np.nan
    x_mean = np.nanmean(X, axis=0)[np.newaxis, :]
    X_no_nan = np.nan_to_num(X, nan=0)
    #x_mean = X_no_nan.mean(axis=0)[np.newaxis, :]
    X -=x_mean
    x_std = np.nanstd(X,axis=0)[np.newaxis, :]
    X /=x_std
    # Perform initial PCA fit
    pca = PCA(n_components=comps) # set max number of components
    pca.fit(X_no_nan)
    X_no_nan[np.isnan(X_no_nan) | np.isinf(X_no_nan)] = 0
    pca_x =  pca.transform(X_no_nan)
    print(pca_x)
    pca_x = pca_x.reshape((dim1, dim2,comps))
    print(pca_x.shape)
    return pca_x

# Store map info for raster
def store_metadata(neon, epsg):
    """Store metadata to create corrected raster with same metadata as original.
    
    Parameters:
    -----------
    neon: hytools image file
    Returns:
    -----------
    refl_md: metadata.
    header_dict: header.
    
    """
    mapInfo= neon.map_info
    header_dict = neon.get_header()
    print(header_dict)
    refl_md = {}
    refl_md['mapInfo'] = header_dict['map info']
    refl_md['wavelength'] = header_dict['wavelength']
    refl_md['shape'] = [neon.lines, neon.columns, neon.bands]
    #Extract no data value & scale factor
    refl_md['noDataVal'] = float(header_dict['data ignore value'])
    refl_md['scaleFactor'] = float(0.996)
    refl_md['bad_band_window1'] = np.array([1340, 1445])
    refl_md['bad_band_window2'] = np.array([1790, 1955])
    #refl_md['epsg'] = 32614 # for wgs 84, UTM 14 for KONZ --> note that this changed by site!!
    refl_md['epsg'] = epsg
    refl_md['res'] = {}
    refl_md['res']['pixelWidth'] = float(mapInfo[5])
    refl_md['res']['pixelHeight'] = float(mapInfo[6])
    # Extract the upper left-hand corner coordinates from mapInfo
    xMin = float(mapInfo[3])  # convert from string to floating point number
    yMax = float(mapInfo[4])
    # Calculate the xMax and yMin values from the dimensions
    xMax = xMin + (refl_md['shape'][1] * refl_md['res']['pixelWidth'])  # xMax = left edge + (# of columns * resolution)",
    yMin = yMax - (refl_md['shape'][0] * refl_md['res']['pixelHeight'])  # yMin = top edge - (# of rows * resolution)",
    refl_md['extent'] = (xMin, xMax, yMin, yMax)  # useful format for plotting
    refl_md['ext_dict'] = {}
    refl_md['ext_dict']['xMin'] = xMin
    refl_md['ext_dict']['xMax'] = xMax
    refl_md['ext_dict']['yMin'] = yMin
    refl_md['ext_dict']['yMax'] = yMax
    print(refl_md)
    return refl_md, header_dict

# Compute CHV --> In progress

## 4. Visualization Tools ####

# Show RGB representation of hytools object.
def show_rgb(hy_obj,r=660,g=550,b=440, correct= []):
    """Display raster in RGB.
    
    Parameters:
    -----------
    x: hytools object
        Hytools object of image of interest.
    r: int
        Wavelength/band corresponding to red.
    g: int
        Wavelength/band corresponding to green.
    b: int
        Wavelength/band corresponding to blue.
    Returns:
    -----------
    rgb : raster
        RGB raster for display in plot viewer.
    
    Plots rgb raster.
    
    """
    rgb=  np.stack([hy_obj.get_wave(r,corrections= correct),
                    hy_obj.get_wave(g,corrections= correct),
                    hy_obj.get_wave(b,corrections= correct)])
    rgb = np.moveaxis(rgb,0,-1).astype(float)
    rgb[rgb ==hy_obj.no_data] = np.nan

    bottom = np.nanpercentile(rgb,5,axis = (0,1))
    top = np.nanpercentile(rgb,95,axis = (0,1))
    rgb = np.clip(rgb,bottom,top)

    rgb = (rgb-np.nanmin(rgb,axis=(0,1)))/(np.nanmax(rgb,axis= (0,1))-np.nanmin(rgb,axis= (0,1)))

   # height = int(hy_obj.lines/hy_obj.columns)

    # fig  = plt.figure(figsize = (7,7) )
    plt.imshow(rgb)
    plt.show()
    #plt.close()
