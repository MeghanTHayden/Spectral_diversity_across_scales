#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 10:25:17 2023

@author: meha3816
"""
## Load necessary packages ##
import hytools as ht
import matplotlib.pyplot as plt
import numpy as np
import requests
import sklearn
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import kneed
from kneed import KneeLocator
import scipy.spatial
from scipy.spatial import ConvexHull
import subprocess
from urllib.request import urlretrieve
import parmap
import os
from tqdm import tqdm
import csv
from csv import writer

def window_calcs_old(args):
    
    """ Calculate convex hull volume for a single PCA chunk and window size.
    FOR USE IN PARALLEL PROCESSING OF FUNCTIONAL RICHNESS.
    
    Parameters:
    -----------
    pca_chunk: PCA chunk from NEON image.
    window_sizes: list/array of integers
    comps: Number of PCs. Here, set to 4. 
    
    Returns:
    -----------
    volume_mean: functional richness for given window size and image.
    
    """
    windows, pca_chunk, results_FR, local_file_path  = args
    #print(pca_chunk[15:20, 30:40, 1])
    window_data = []
    for window in tqdm(windows, desc='Processing window for batch'):
        comps = 3
        half_window = window // 2
        #print(pca_chunk.shape)
        fric = np.zeros((pca_chunk.shape[0], pca_chunk.shape[1]))
        for i in tqdm(range(half_window, pca_chunk.shape[0] - half_window, 25), desc='Processing window index'):
            for j in range(half_window, pca_chunk.shape[1] - half_window, 25):
                hull = None
                sub_arr = pca_chunk[i - half_window:i + half_window + 1, j - half_window:j + half_window + 1, :]
                #print(sub_arr.shape)
                sub_arr = sub_arr.reshape((-1, comps))
                #print(i,j)
                #print(sub_arr.shape)
                mean_arr = np.nanmean(sub_arr, axis=0)
                #print(mean_arr)
                non_zero_indices = np.nonzero(mean_arr)[0]
                print(non_zero_indices)
                # Try something to increase efficiency
                #unique_points = np.unique(sub_arr, axis=0)
                #min_bounds = sub_arr.min(axis=0)
                #max_bounds = sub_arr.max(axis=0)
                # Keep points close to the edges of the bounding box
                #buffer = 1e-6  # Small buffer to include near-boundary points
                #filtered_points = data[
                #    (sub_arr[:, 0] <= min_bounds[0] + buffer) | (sub_arr[:, 0] >= max_bounds[0] - buffer) |
                #    (sub_arr[:, 1] <= min_bounds[1] + buffer) | (sub_arr[:, 1] >= max_bounds[1] - buffer)
                #]
                # Continue as normal
                if len(non_zero_indices) >= 3:
                    try:
                        if hull is None:
                            #print(sub_arr[:,non_zero_indices])
                            hull = ConvexHull(sub_arr)
                        #fric[i, j] = hull.volume
                        #print(hull.volume)
                        #print(fric)
                    except scipy.spatial.qhull.QhullError as e:
                        continue
                window_data.append([window, hull.volume])
        print(f"Hull volumes for window size {window}: {np.unique(fric)}")
        #results_FR[window] = fric.tolist()

        with open(local_file_path, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            if csvfile.tell() == 0:
                csvwriter.writerow(['Window_Size', 'Hull_Volume'])  # Write header
                                        
           # for window_result, fric_matrix in results_FR.items():
           #     for row in fric_matrix:
           #         csvwriter.writerow([window_result] + row)
            for data_point in window_data:
                csvwriter.writerow(data_point)
        #results_FR.append(np.nanmean(fric))
    return results_FR

def window_calcs(args):
    """
    Calculate convex hull-based functional richness (volume) for a PCA chunk
    across one or more window sizes.

    Parameters
    ----------
    args : tuple
        (windows, pca_chunk, results_FR, local_file_path)

        windows        : list/array of integer window sizes (odd integers)
        pca_chunk      : 3D array (rows, cols, n_pc) of PC scores
        results_FR     : dict or list for storing results (optional)
        local_file_path: path to CSV file for writing results

    Returns
    -------
    results_FR : dict
        Dictionary keyed by window size with a list of [row, col, hull_volume]
    """
    windows, pca_chunk, results_FR, local_file_path = args

    # Ensure numpy array and unpack dimensions
    pca_chunk = np.asarray(pca_chunk)
    n_rows, n_cols, n_pc = pca_chunk.shape

    # Use up to 5 PCs (or fewer if the array has fewer than 5)
    comps = min(5, n_pc)

    batch_results = {}

    # Open the CSV once per batch
    with open(local_file_path, "a", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        # Write header only if file is empty
        if csvfile.tell() == 0:
            csvwriter.writerow(["Window_Size", "Row", "Col", "Hull_Volume"])

        for window in tqdm(windows, desc="Processing window size for batch"):
            half_window = window // 2
            window_data = []  # results for this specific window

            # Stride = 50 (up from 25 in previous implementations)
            for i in tqdm(
                range(half_window, n_rows - half_window, 50),
                desc=f"Processing indices for window {window}",
                leave=False
            ):
                for j in range(half_window, n_cols - half_window, 50):
                    # Extract subwindow: (window x window x n_pc)
                    sub_arr = pca_chunk[
                        i - half_window:i + half_window + 1,
                        j - half_window:j + half_window + 1,
                        :comps  # first 'comps' PCs
                    ]

                    # Reshape to (n_points, comps)
                    sub_arr = sub_arr.reshape(-1, comps)

                    # Drop rows with any NaNs
                    valid = ~np.isnan(sub_arr).any(axis=1)
                    sub_arr_valid = sub_arr[valid, :]

                    # Need at least comps + 1 unique points for a hull in 'comps' dims
                    if sub_arr_valid.shape[0] < comps + 1:
                        continue

                    # Remove duplicate points to reduce hull complexity
                    unique_points = np.unique(sub_arr_valid, axis=0)
                    if unique_points.shape[0] < comps + 1:
                        continue

                    # For now use all unique points.
                    points_for_hull = unique_points

                    try:
                        hull = ConvexHull(points_for_hull)
                        hull_volume = hull.volume
                    except scipy.spatial.qhull.QhullError:
                        # Degenerate case: points not in general position in 5D
                        continue

                    window_data.append([window, i, j, hull_volume])
                    csvwriter.writerow([window, i, j, hull_volume])

            batch_results[window] = window_data

    # Merge into results_FR if it's a dict, otherwise just return batch_results
    if isinstance(results_FR, dict):
        results_FR.update(batch_results)
        return results_FR
    else:
        return batch_results

def wave_calcs(args):
    wave, neon = args
    arrays = neon.get_wave(wave, corrections= ['topo','brdf'])
    return arrays
        
