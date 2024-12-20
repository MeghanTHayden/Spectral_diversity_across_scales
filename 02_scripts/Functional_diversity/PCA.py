#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Dec 20 2024

@author: meha3816
"""

import os
import numpy as np
import rasterio
from sklearn.decomposition import PCA
import pandas as pd

def find_rasters(SITECODE, bucket_name):
  
  # List mosaics for the site of interest
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

  return plots
  
def load_rasters(SITECODE, plot, bucket_name):
  # Load data
  file_stem = SITECODE + '_flightlines/Mosaic_' + SITECODE + '_'
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

  return veg_np

def perform_pca(data, n_components=3):
    
    # Flatten features into one dimension
    shape = data.shape
    print("Raster shape:", shape)
    dim1, dim2, bands = shape[1], shape[2], shape[0]
    X = data.reshape(bands, dim1 * dim2).T
    print("Shape of flattened array:", X.shape)

    # Set no data to nan
    X = X.astype('float32')
    X[np.isnan(X)] = np.nan
    X[X <= 0] = np.nan  # Adjust threshold if needed
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
    
    return pca_x

def write_pca_to_raster(SITECODE, Data_Dir, Out_Dir, pca_x):
    # Define the output raster file name
    local_pca_raster_path = os.path.join(Out_Dir, f"{SITECODE}_pca_{i}.tif")
    s3_pca_raster_key = f"/{SITECODE}_pca_{i}.tif"

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

def process_and_write_pca(SITECODE, Data_Dir):
  # Initialize S3
  bucket_name = 'bioscape.gra'
  s3 = boto3.client('s3')
  # Find mosaics for site of interest
  plots = find_rasters(SITECODE, Data_Dir, bucket_name)
  # Load and process PCA, write to S3
  for plot in plots:
    veg_np = load_rasters(SITECODE, plot, bucket_name)
    pca_x = process_pca(veg_np, n_components = 3)
    write_pca_to_raster(SITECODE, Data_Dir, Out_Dir, pca_x)

# Example usage
if __name__ == "__main__":
    
    process_and_write_pca(SITECODE, Data_Dir, Out_Dir)
