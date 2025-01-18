# Load required libraries
import os
import h5py
import rasterio
from rasterio.transform import from_origin
import numpy as np
import requests
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError

def calculate_bounding_box(center_lat, center_lon, box_size_km=2):
    # Convert box size to degrees (1 km ~ 0.008998 degrees latitude/longitude)
    offset = (box_size_km / 2) * 0.008998
    return {
        "min_lat": center_lat - offset,
        "max_lat": center_lat + offset,
        "min_lon": center_lon - offset,
        "max_lon": center_lon + offset
    }

def search_neon_products(bbox, product_code):
    base_url = "https://data.neonscience.org/api/v0/products"
    
    # NEON API request parameters
    params = {
        "bbox": f"{bbox['min_lon']},{bbox['min_lat']},{bbox['max_lon']},{bbox['max_lat']}",
        "productCode": product_code
    }
    
    # Make the API request
    response = requests.get(base_url, params=params)
    response.raise_for_status()
    data = response.json()
    
    # Parse and return available data
    available_data = data.get("data", {}).get("availableData", [])
    return available_data

def download_neon_product(product, download_path):
    for file_url in product.get("files", []):
        file_name = file_url.split("/")[-1]
        file_response = requests.get(file_url, stream=True)
        file_response.raise_for_status()
        
        with open(f"{download_path}/{file_name}", "wb") as file:
            for chunk in file_response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"Downloaded {file_name}")
    return file_name

def inspect_h5_metadata(h5_file):
    """
    Inspect the structure and metadata of an H5 file.
    """
    with h5py.File(h5_file, 'r') as h5:
        print("Keys in the file:")
        def print_structure(name, obj):
            if isinstance(obj, h5py.Group):
                print(f"Group: {name}")
            elif isinstance(obj, h5py.Dataset):
                print(f"Dataset: {name}")
                print(f" - Shape: {obj.shape}")
                print(f" - Data Type: {obj.dtype}")
                for key, value in obj.attrs.items():
                    print(f" - Attribute: {key} = {value}")
        
        h5.visititems(print_structure)

def process_h5_to_raster(h5_file, raster_path):
    """
    Convert an H5 file to a GeoTIFF raster.
    """
    with h5py.File(h5_file, 'r') as h5:
        # Assuming you know the dataset structure in the H5 file
        # Modify this path to match the dataset you want to extract
        dataset = h5['/Data/Values']  # Example path
        data = dataset[:]
        
        # Retrieve metadata (e.g., geotransform, CRS)
        # Adjust these values based on your .h5 file structure
        transform = from_origin(0, 0, 1, 1)  # Dummy transform (adjust as needed)
        crs = "EPSG:4326"  # Replace with the correct CRS
        
        # Write the data to a GeoTIFF raster
        with rasterio.open(
            raster_path,
            'w',
            driver='GTiff',
            height=data.shape[0],
            width=data.shape[1],
            count=1,
            dtype=data.dtype,
            crs=crs,
            transform=transform
        ) as dst:
            dst.write(data, 1)

    print(f"Raster saved at {raster_path}")

def upload_to_s3(local_file, bucket_name, s3_key):
    """
    Upload a file to S3.
    """
    s3 = boto3.client('s3')
    try:
        s3.upload_file(local_file, bucket_name, s3_key)
        print(f"Uploaded {local_file} to s3://{bucket_name}/{s3_key}")
    except (NoCredentialsError, PartialCredentialsError) as e:
        print(f"Error: {e}")

def process_and_upload(h5_file, Out_Dir, bucket_name):
    """
    Convert H5 to raster and upload to S3.
    """
    # Generate raster file path
    raster_name = os.path.basename(h5_file).replace(".h5", ".tif")
    raster_path = os.path.join(Out_Dir, raster_name)
    
    # Process H5 to raster
    process_h5_to_raster(h5_file, raster_path)
    
    # Upload to S3
    s3_key = f"new_raster_outputs/{raster_name}"
    upload_to_s3(raster_path, bucket_name, s3_key)

def neon_flight_process(center_lat, center_long):
  Data_Dir = '/home/ec2-user/BioSCape_across_scales/01_data/'
  Out_Dir = '/home/ec2-user/BioSCape_across_scales/02_output/'
  bucket_name = 'bioscape.gra'
  s3 = boto3.client('s3')
  
  bbox = calculate_bounding_box(center_lat, center_lon)
  
  # Search for DP3.30006.002 products
  print("Searching for overlapping mosaics...")
  products = search_neon_products(bbox, "DP3.30006.002")
  print(products)

  for product in products:
    print("Downloading mosaics...")
    download_neon_product(product, Data_Dir)

  for h5_file in os.listdir(Data_Dir):
        if h5_file.endswith(".h5"):
            h5_file_path = os.path.join(Data_Dir, h5_file)
            inspect_h5_metadata(h5_file)
            #print(f"Uploading {h5_file_path} as raster to s3")
            #process_and_upload(h5_file_path, Out_Dir, bucket_name)
            #os.remove(h5_file_path)

center_lat = 28.07484
center_lon = -81.39467
neon_flight_process(center_lat, center_lon)

