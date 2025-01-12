#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 09:25:54 2024

@author: meha3816
"""
import os
import re
import gc
import boto3
import csv
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ClientError
import rasterio
import rioxarray as rxr
import numpy as np
import sklearn
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import IncrementalPCA
import requests
from multiprocessing import Pool
from sklearn.pipeline import Pipeline

def identify_plots(SITECODE, s3, bucket_name):
  # List mosaics for a site in the S3 bucket in the matching directory
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

def load_data_and_mask(SITECODE, plot, s3, bucket_name, Data_Dir):
  print("Loading plot:", plot)
  # Load data
  file_stem = SITECODE + '_flightlines/Mosaic_' + SITECODE + '_'
  clip_file = file_stem + str(plot) + '.tif' # Define file name in S3
  print(clip_file)
  
  # Download plot mosaic
  local_file = f"{Data_Dir}/mosaic_{plot}.tif"
  s3.download_file(bucket_name, clip_file, local_file)
  print(f"Raster downloaded to {local_file}")
  
  # Open as raster
  raster = rxr.open_rasterio(local_file, masked=True)
  crs = raster.rio.crs
  transform = raster.rio.transform()
  print(raster)
  
  # Convert data array to numpy array
  veg_np = raster.to_numpy()
  shape = veg_np.shape
  print("Raster shape:", shape)

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
  prop_na = np.isnan(X).mean()
  print("Proportion of NaN values:", prop_na)

  # Rescale data
  X /= 10000

  # Remove unnecessary files
  os.remove(local_file)
  del veg_np

  return X, nan_mask, prop_na, dim1, dim2, crs, transform

def perform_pca(X, nan_mask, dim1, dim2, ncomps = 3):
  # Impute missing values
  print("Imputing missing values...")
  #imputer = SimpleImputer(missing_values=np.nan, strategy='median')
  #X_transformed = imputer.fit_transform(X)

  # Scale & standardize array
  print("Scaling data...")
  #scaler = RobustScaler()
  #X_transformed = scaler.fit_transform(X_transformed)
  
  # Combine in one step
  pipeline = Pipeline([
          ('imputer', SimpleImputer(missing_values = np.nan,strategy='median')),
              ('scaler', RobustScaler())
              ])

  X_transformed = pipeline.fit_transform(X)

  # Perform initial PCA fit
  print("Fitting PCA")
  pca = PCA(n_components=ncomps)
  pca.fit(X_transformed)
  var_explained = pca.explained_variance_ratio_
  print("Explained variance ratio:", var_explained)

  # PCA transform
  pca_x = pca.transform(X_transformed)
  pca_x = pca_x.reshape((dim1, dim2, ncomps))
  print("PCA shape:", pca_x.shape)

  # Reapply NaN mask
  nan_mask_reshaped = nan_mask[:, 0].reshape(dim1, dim2)  # Ensure mask matches raster dimensions
  for i in range(ncomps):  # Apply mask to each PCA component
      pca_x[:, :, i][nan_mask_reshaped] = np.nan

  return pca_x, var_explained

def perform_pca_chunks(X, nan_mask, dim1, dim2, ncomps = 3):
  chunk_size = 10000
  ipca = IncrementalPCA(n_components=ncomps)
  pipeline = Pipeline([
          ('imputer', SimpleImputer(missing_values = np.nan,strategy='median')),
              ('scaler', RobustScaler())
              ])
  # Fitting IPCA
  print("Fitting IPCA")
  for start in range(0, len(X), chunk_size):
    end = start + chunk_size
    X_chunk = X[start:end]
    if X_chunk.shape[1] == 0:
        print(f"Chunk with shape {X_chunk.shape} has no valid features. Skipping.")
        continue
    X_chunk_transformed = pipeline.fit_transform(X_chunk)
    ipca.partial_fit(X_chunk_transformed)
  # Tranforming IPCA
  print("Transforming IPCA")
  pca_results = []
  for start in range(0, len(X), chunk_size):
    end = start + chunk_size
    X_chunk = X[start:end]
    if X_chunk.shape[1] == 0:
        print(f"Chunk with shape {X_chunk.shape} has no valid features. Skipping.")
        continue
    X_chunk_transformed = pipeline.transform(X_chunk)
    chunk_pca = ipca.transform(X_chunk_transformed)
    pca_results.append(chunk_pca)

  # Combine PCA-transformed chunks
  combined_pca_results = np.vstack(pca_results)

  # Reshape PCA-transformed data to match original dimensions
  pca_reshaped = combined_pca_results.reshape(dim1, dim2, ncomps)

  # Reapply NaN mask to each PCA component
  nan_mask_reshaped = nan_mask[:, 0].reshape(dim1, dim2)
  for i in range(ncomps):
      pca_reshaped[:, :, i][nan_mask_reshaped] = np.nan

  # Return PCA results and variance explained
  print("PCA processing complete")
  print(f"PCA explained variance:{ipca_explained_variance_ratio_}")
  return pca_reshaped, ipca.explained_variance_ratio_

def write_pca_to_raster(SITECODE, Data_Dir, Out_Dir, s3, bucket_name, pca_x, plot, crs, transform):
    # Define the output raster file name
    local_pca_raster_path = os.path.join(Out_Dir, f"{SITECODE}_pca_{plot}.tif")
    s3_pca_raster_key = f"{SITECODE}_flightlines/{SITECODE}_pca_{plot}.tif"

    # Write PCA to raster
    with rasterio.open(
        local_pca_raster_path,
        "w",
        driver="GTiff",
        height=pca_x.shape[0],
        width=pca_x.shape[1],
        count=pca_x.shape[2],  # Number of components as raster bands
        dtype="float32",
        crs=crs,  # Use CRS from the original raster
        transform=transform,  # Use transform from the original raster
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

def write_plot_variables_to_csv(Out_Dir, SITECODE, results,s3, bucket_name):
  # Export results to a CSV
  local_csv_path = os.path.join(Out_Dir, f"{SITECODE}_pca_summary.csv")
  with open(local_csv_path, mode='w', newline='') as csvfile:
      writer = csv.DictWriter(csvfile, fieldnames=["SITECODE", "plot", "Proportion_NaN", "Explained_Variance"])
      writer.writeheader()
      for row in results:
          writer.writerow(row)

  # Upload the file to S3
  s3_csv_key = f"{SITECODE}_flightlines/{SITECODE}_pca_summary.csv"
  try:
      s3.upload_file(local_csv_path, bucket_name, s3_csv_key)
      print(f"PCA csv uploaded to S3: {s3_csv_key}")
  except ClientError as e:
      print(f"Error uploading csv to S3: {e}")


def parallel_pca_workflow(SITECODE):
    # Define variables
    Data_Dir = '/home/ec2-user/Functional_diversity_across_scales/01_data'
    Out_Dir = '/home/ec2-user/Functional_diversity_across_scales/02_output'
    bucket_name = 'bioscape.gra'
    s3 = boto3.client('s3')

    # Get list of plots
    plots = identify_plots(SITECODE, s3, bucket_name)

    # Prepare arguments for multiprocessing
    args = [(SITECODE, plot, Data_Dir, Out_Dir, bucket_name) for plot in plots]

    # Process plots in parallel
    with Pool(processes=3) as pool:  # Adjust the number of processes based on available CPUs
                results = pool.starmap(process_plot, args)

    # Filter out any failed results
    results = [res for res in results if res is not None]

    # Write results to csv
    s3 = boto3.client('s3')
    write_plot_variables_to_csv(Out_Dir, SITECODE, results, s3, bucket_name)

    print(f"PCA processing complete for {SITECODE}")


def pca_workflow(SITECODE):
  # Set directories
  Data_Dir = '/home/ec2-user/Functional_diversity_across_scales/01_data'
  Out_Dir = '/home/ec2-user/Functional_diversity_across_scales/02_output'
  bucket_name = 'bioscape.gra'
  s3 = boto3.client('s3')

  # List to store plot level variables
  results = []
  
  plots = identify_plots(SITECODE, s3, bucket_name)
  for plot in plots:
    X, nan_mask, prop_na,dim1,dim2,crs,transform = load_data_and_mask(SITECODE, plot, s3, bucket_name, Data_Dir)
    pca_x, var_explained = perform_pca(X, nan_mask, dim1, dim2, ncomps = 3)
    results.append({
        "SITECODE": SITECODE,
        "plot": plot,
        "Proportion_NaN": prop_na,
        "Explained_Variance": var_explained.tolist()  # Convert to list for CSV
    })
    write_pca_to_raster(SITECODE, Data_Dir, Out_Dir, s3, bucket_name, pca_x, plot, crs, transform)
   
    # Clear unnecessary variables from memory
    del X, nan_mask, prop_na, pca_x, var_explained
    gc.collect()  # Trigger garbage collection
  
  write_plot_variables_to_csv(Out_Dir, SITECODE, results, s3, bucket_name)
  print(f"PCA processing complete for {SITECODE}")
 
# Example usage
if __name__ == "__main__":

    pca_workflow(SITECODE)
   #parallel_pca_workflow(SITECODE)
    
