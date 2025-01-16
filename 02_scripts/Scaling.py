import os
import boto3
import pandas as pd
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ClientError

def identify_summaries(s3, bucket_name, search_criteria):
    """
    Identify all CSV files in the S3 bucket matching the search criteria.
    """
    try:
        # List objects in the S3 bucket
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix="/")
        objects = response.get('Contents', [])
        
        # Filter objects based on the search criteria
        summaries = [{"Key": obj['Key'], "LastModified": obj['LastModified']}
                     for obj in objects if obj['Key'].endswith('.csv') and search_criteria in obj['Key']]
        print(f"Found {len(summaries)} matching files.")
        return summaries

    except ClientError as e:
        print(f"Error accessing S3 bucket: {e}")
        return []

def process_csv_files(s3, bucket_name, summaries, output_csv):
    """
    Download, process, and compute summary statistics for each CSV file.
    """
    results = []  # To store results
    
    for summary_file in summaries:
        summary_file = summary["Key"]
        last_modified = summary["LastModified"]
        print(f"Processing file: {summary_file} (Last Modified: {last_modified})")
        try:
            # Download the file from S3
            local_file = os.path.basename(summary_file)
            s3.download_file(bucket_name, summary_file, local_file)

            # Load the CSV into a pandas DataFrame
            df = pd.read_csv(local_file)
            
            # Compute summary statistics (e.g., mean specdiv for each window size)
            if 'Window_Size' in df.columns and 'Hull_Volume' in df.columns:
                summary_stats = df.groupby('Window_Size')['Hull_Volume'].median().reset_index()
                summary_stats['File_Name'] = summary_file
                summary_stats['Last_Modified'] = last_modified  # Add last modified date
                results.append(summary_stats)
            else:
                print(f"Required columns not found in {summary_file}")

            # Clean up the local file
            os.remove(local_file)

        except ClientError as e:
            print(f"Error downloading or processing {summary_file}: {e}")

    # Combine all results into a single DataFrame
    if results:
        combined_results = pd.concat(results, ignore_index=True)
        
        # Save results to a CSV file
        combined_results.to_csv(output_csv, index=False)
        print(f"Summary results saved to {output_csv}")
        
        return output_csv  # Return the file path for uploading
    else:
        print("No results to save.")
        return None

def upload_results_to_s3(s3, bucket_name, local_file_path, destination_key):
    """
    Upload the results file to the S3 bucket.
    """
    try:
        print(f"Uploading {local_file_path} to S3 bucket {bucket_name}...")
        s3.upload_file(local_file_path, bucket_name, destination_key)
        print(f"File uploaded to S3: {destination_key}")
    except ClientError as e:
        print(f"Error uploading file to S3: {e}")

def main():
    bucket_name = 'bioscape.gra'
    search_criteria = "specdiv"
    output_csv = "results_summary.csv"
    destination_key = "/Specdiv_results_summary_1-16.csv"  # S3 path for the uploaded file

    # Initialize the S3 client
    s3 = boto3.client('s3')

    # Step 1: Identify relevant CSV files
    summaries = identify_summaries(s3, bucket_name, search_criteria)

    # Step 2: Process each file and compute summary statistics
    if summaries:
        local_file_path = process_csv_files(s3, bucket_name, summaries, output_csv)
        if local_file_path:
            # Step 3: Upload the results file to S3
            upload_results_to_s3(s3, bucket_name, local_file_path, destination_key)
    else:
        print("No matching files found in the S3 bucket.")

if __name__ == "__main__":
    main()
