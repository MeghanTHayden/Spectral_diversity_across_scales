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
from multiprocessing import Pool, cpu_count
from sklearn.pipeline import Pipeline
from tqdm.contrib.concurrent import process_map
from S01_Moving_Window_FRIC import *
from S01_Moving_Window_FDiv import *

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
  #nan_mask = np.isnan(X)
  prop_na = np.isnan(X).mean()
  print("Proportion of NaN values:", prop_na)

  # Rescale data
  X /= 10000

  # Remove unnecessary files
  os.remove(local_file)
  del veg_np

  return X, prop_na, dim1, dim2, crs, transform

def perform_pca(X, dim1, dim2, ncomps = 3):
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

  return pca_x, var_explained

def randomize_pixels(pca_x):
  # Randomize pixels to remove spatial organization
  print("Randomizing pixels for null distribution...")
  shape = pca_x.shape
  dim1, dim2, ncomps = shape[0], shape[1], shape[2]
  pca_x_flat = pca_x.reshape(-1, ncomps)  # Flatten PCA array to 2D (n_pixels, n_components)
  print("Flattened shape for randomization:", pca_x_flat.shape)
  pca_x_flat_shuffled = np.random.permutation(pca_x_flat)          # Shuffle rows randomly
  pca_x_random = pca_x_flat_shuffled.reshape(dim1, dim2,ncomps)  # Reshape back to original dimensions
  print("Randomization complete. Shape:", pca_x_random.shape)

  # Verify randomization
  print(f"Original data sample: {pca_x[200:210]}")
  print(f"Randomized data sample: {pca_x_random[200:210]}")

  return pca_x_random

def calculate_fric(SITECODE, plot, pca_x, window_sizes, bucket_name, Out_Dir):
  # Calculate FRic on PCA across window sizes
  print("Calculating FRic")
  results_FR = {}
  local_file_path_fric = Out_Dir + "/" + SITECODE + "_fric_" + str(plot) + ".csv"
  window_batches = [(a, pca_x, results_FR, local_file_path_fric) for a in np.array_split(window_sizes, cpu_count() - 1) if a.any()]
  volumes = process_map(
       window_calcs,
       window_batches,
       max_workers=cpu_count() - 1
   )
  destination_s3_key_fric = "/" + SITECODE + "_specdiv_" + str(plot) + ".csv"
  s3 = boto3.client('s3')
  s3.upload_file(local_file_path_fric, bucket_name, destination_s3_key_fric)
  print("FRic file uploaded to S3")

def calculate_fric_null(SITECODE, plot, pca_x_random, window_sizes, bucket_name, Out_Dir):
  # Calculate FRic on PCA across window sizes
  print("Calculating FRic")
  results_FR_null = {}
  local_file_path_fric_null = Out_Dir + "/" + SITECODE + "_fric_null_" + str(plot) + ".csv"
  window_batches = [(a, pca_x_random, results_FR_null, local_file_path_fric_null) for a in np.array_split(window_sizes, cpu_count() - 1) if a.any()]
  volumes = process_map(
       window_calcs,
       window_batches,
       max_workers=cpu_count() - 1
   )
  destination_s3_key_fric = "/" + SITECODE + "_specdiv_null_" + str(plot) + ".csv"
  s3 = boto3.client('s3')
  s3.upload_file(local_file_path_fric_null, bucket_name, destination_s3_key_fric)
  print("Null FRic file uploaded to S3")

def calculate_fdiv_null(SITECODE, plot, pca_x_random, window_sizes, bucket_name, Out_Dir):
  results_FD_null = {}
  local_file_path_fdiv_null = Out_Dir + "/" + SITECODE + "_fdiv_veg_" + str(plot) + ".csv"
  window_batches = [(a, pca_x_random, results_FD_null, local_file_path_fdiv_null) for a in np.array_split(window_sizes, cpu_count() - 1) if a.any()]
  volumes = process_map(
    window_calcs_fdiv,
    window_batches,
    max_workers=cpu_count() - 1
  )
  # open file for writing
  destination_s3_key_fdiv_null = "/" + SITECODE + "_specdivergence_null_" + str(plot) + ".csv"
  s3 = boto3.client('s3')
  s3.upload_file(local_file_path_fdiv_null, bucket_name,destination_s3_key_fdiv_null)
  print("FDiv file uploaded to S3")

def pca_specdiv_workflow(SITECODE):
  # Set directories
  # Depends on instance
  #Data_Dir = '/home/ec2-user/Functional_diversity_across_scales/01_data'
  #Out_Dir = '/home/ec2-user/Functional_diversity_across_scales/02_output'
  Data_Dir = '/home/ec2-user/BioSCape_across_scales/01_data'
  Out_Dir = '/home/ec2-user/BioSCape_across_scales/02_output'
  bucket_name = 'bioscape.gra'
  s3 = boto3.client('s3')
  window_sizes = [60, 120, 240, 480, 960, 1200, 1500, 2000, 2200]
  
  plots = identify_plots(SITECODE, s3, bucket_name)
  for plot in plots:
    X, prop_na,dim1,dim2,crs,transform = load_data_and_mask(SITECODE, plot, s3, bucket_name, Data_Dir)
    pca_x, var_explained = perform_pca(X, dim1, dim2, ncomps = 3)
    pca_x_random = randomize_pixels(pca_x)
    calculate_fric_null(SITECODE, plot, pca_x_random, window_sizes, bucket_name, Out_Dir)
    calculate_fdiv_null(SITECODE, plot, pca_x_random, window_sizes, bucket_name, Out_Dir)
    
    # Clear unnecessary variables from memory
    del X, prop_na, pca_x, var_explained
    gc.collect()  # Trigger garbage collection
  
    print(f"Null specdiv processing complete for {SITECODE}, {plot}")
 
# Example usage
if __name__ == "__main__":

    pca_specdiv_workflow(SITECODE)
