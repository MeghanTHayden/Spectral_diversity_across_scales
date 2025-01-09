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
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ClientError
import rasterio
import rioxarray
import numpy as np
import scikit-learn
import requests

def identify_plots(SITECODE):
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

def load_data_and_mask(plot):
  # Load data
  file_stem = SITECODE + '_flightlines/Mosaic_' + SITECODE + '_'
  clip_file = file_stem + str(plot) + '.tif' # Define file name in S3
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
  os.remove(raster)
  os.remove(veg_np)

  return X, nan_mask, prop_na

def perform_pca(X, nan_mask, ncomps = 3):
  # Impute missing values
  imputer = SimpleImputer(missing_values=np.nan, strategy='median')
  X_transformed = imputer.fit_transform(X)

  # Scale & standardize array
  scaler = RobustScaler()
  X_transformed = scaler.fit_transform(X_transformed)

  # Perform initial PCA fit
  print("Fitting PCA")
  pca = PCA(n_components=ncomps)
  pca.fit(X_transformed)
  var_explained = pca.explained_variance_ratio_
  print("Explained variance ratio:", var_explained)

  # PCA transform
  pca_x = pca.transform(X_transformed)
  pca_x = pca_x.reshape((dim1, dim2, comps))
  print("PCA shape:", pca_x.shape)

  # Reapply NaN mask
  nan_mask_reshaped = nan_mask[:, 0].reshape(dim1, dim2)  # Ensure mask matches raster dimensions
  for i in range(comps):  # Apply mask to each PCA component
      pca_x[:, :, i][nan_mask_reshaped] = np.nan

  return pca_x, var_explained

def write_pca_to_raster(SITECODE, Data_Dir, bucket_name, pca_x, plot):
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

write_plot_variables_to_csv(Out_Dir, SITECODE, results, bucket_name):
  # Export results to a CSV
  local_csv_path = os.path.join(Out_Dir, f"{SITECODE}_pca_summary.csv")
  with open(csv_path, mode='w', newline='') as csvfile:
      writer = csv.DictWriter(csvfile, fieldnames=["SITECODE", "Plot", "Proportion_NaN", "Explained_Variance"])
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

def pca_workflow(SITECODE):
  # Set directories
  Data_Dir = '/home/ec2-user/BioSCape_across_scales/01_data/02_processed'
  Out_Dir = '/home/ec2-user/BioSCape_across_scales/03_output'
  bucket_name = 'bioscape.gra'
  s3 = boto3.client('s3')

  # List to store plot level variables
  results = []
  
  plots = identity_plots(SITECODE)
  for plot in plots:
    X, nan_mask, prop_na = load_data_and_mask(plot)
    pca_x, var_explained = perform_pca(X, nan_mask, ncomps = 3)
    results.append({
        "SITECODE": SITECODE,
        "plot": plot,
        "Proportion_NaN": prop_na,
        "Explained_Variance": var_explained.tolist()  # Convert to list for CSV
    })
    write_pca_to_raster(SITECODE, Data_Dir, bucket_name, pca_x, plot)
    write_plot_variables_to_csv(Out_Dir, SITECODE, results, bucket_name)
    
    # Clear unnecessary variables from memory
    del X, nan_mask, prop_na, pca_x, var_explained
    gc.collect()  # Trigger garbage collection

  print(f"PCA processing complete for {SITECODE}")
 
# Example usage
if __name__ == "__main__":
    
    pca_workflow(SITECODE)
    
