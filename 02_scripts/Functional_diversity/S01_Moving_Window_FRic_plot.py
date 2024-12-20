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

def window_calcs(i, shape, pca_chunk, local_file_path):
    
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
    window_data = []
    fric = np.zeros((pca_chunk.shape[0], pca_chunk.shape[1]))
    comps = 10
    hull = None
    sub_arr = pca_chunk.reshape((-1, comps))
    mean_arr = np.nanmean(sub_arr, axis=0)
    non_zero_indices = np.nonzero(mean_arr)[0]
    hull = ConvexHull(sub_arr)
    window_data.append([str(i), shape, hull.volume])
    print(f"Hull volume for plot {i}: {hull.volume}")

    with open(local_file_path, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        if csvfile.tell() == 0:
            csvwriter.writerow(['plotID','plotSize','Hull_Volume'])  # Write header
                          
        for data_point in window_data:
            csvwriter.writerow(data_point)

    return hull.volume
        
