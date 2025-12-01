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

def compute_global_breaks(pcs_all, n_bins=10, q_low=0.01, q_high=0.99):
    """
    Compute global bin edges for each PC dimension to be reused
    across all windows, based on quantile-trimmed ranges.

    pcs_all: array (N, D) of PC scores or a representative sample.
    n_bins: number of bins per dimension.
    q_low, q_high: quantiles to trim extremes.
    """
    pcs_all = np.asarray(pcs_all)
    n_dims = pcs_all.shape[1]
    breaks_list = []

    for d in range(n_dims):
        low = np.nanquantile(pcs_all[:, d], q_low)
        high = np.nanquantile(pcs_all[:, d], q_high)
        if high <= low:
            high = low + 1e-6
        edges = np.linspace(low, high, n_bins + 1)
        breaks_list.append(edges)

    return breaks_list


def compute_tpd_histogram(Xw, breaks_list):
    """
    Compute D-dimensional histogram-based TPD for one window.

    Xw: (n_w, D) array of PC scores for this window.
    breaks_list: list of bin edges per dimension.
    """
    Xw = np.asarray(Xw)
    if Xw.size == 0:
        dims = [len(b) - 1 for b in breaks_list]
        return np.zeros(dims, dtype=float)

    hist, _ = np.histogramdd(Xw, bins=breaks_list)
    total = hist.sum()
    if total == 0:
        return np.zeros_like(hist, dtype=float)

    prob = hist / total
    return prob, hist


def tpd_richness(prob, hist, min_count = 2):
    """Number of occupied cells (prob > threshold)."""
    return float(np.count_nonzero(hist >= min_count))


def tpd_entropy(prob, eps=1e-12):
    """Shannon entropy (nats) of TPD."""
    p = prob[prob > 0]
    if p.size == 0:
        return 0.0
    return float(-(p * np.log(p + eps)).sum())

def window_calcs(args):
    """
    Calculate TPD-based functional richness for a single PCA chunk and window size(s).

    Parameters
    ----------
    args : tuple
        (windows, pca_chunk, breaks_list, local_file_path)

        windows        : list/array of integer window sizes (odd integers)
        pca_chunk      : 3D array (rows, cols, n_pc) of PC scores
        breaks_list    : list of bin edges for each PC dimension (from compute_global_breaks)
        local_file_path: path to CSV file for writing results

    Returns
    -------
    results_FR : dict
        Dictionary keyed by window size with a list of [row, col, richness, entropy]
        for each evaluated window location (optional – here not heavily used).
    """
    windows, pca_chunk, breaks_list, local_file_path = args

    # Make sure pca_chunk is a numpy array
    pca_chunk = np.asarray(pca_chunk)
    n_rows, n_cols, n_pc = pca_chunk.shape

    results_FR = {}
    # Open CSV once per call to avoid reopening for each window
    # We'll append rows for all windows in this chunk.
    with open(local_file_path, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # If file is empty, write header
        if csvfile.tell() == 0:
            csvwriter.writerow(['Window_Size', 'Row', 'Col',
                                'TPD_Richness', 'TPD_Entropy'])

        for window in tqdm(windows, desc='Processing window size for chunk'):
            half_window = window // 2
            window_results = []

            # Loop over center positions with step size 25 (as in your original code)
            for i in tqdm(range(half_window, n_rows - half_window, 25),
                          desc=f'Processing indices for window {window}', leave=False):
                for j in range(half_window, n_cols - half_window, 25):
                    # Extract subwindow: (window x window x n_pc)
                    sub_arr = pca_chunk[
                        i - half_window:i + half_window + 1,
                        j - half_window:j + half_window + 1,
                        :
                    ]

                    # Reshape to (n_w, n_pc)
                    sub_arr = sub_arr.reshape(-1, n_pc)

                    # Optionally drop rows with all NaNs
                    valid = ~np.isnan(sub_arr).any(axis=1)
                    sub_arr_valid = sub_arr[valid, :]

                    if sub_arr_valid.shape[0] == 0:
                        # No valid pixels in this window
                        richness = np.nan
                        entropy_val = np.nan
                    else:
                        # Compute TPD histogram and metrics
                        prob, hist = compute_tpd_histogram(sub_arr_valid, breaks_list)
                        richness = tpd_richness(prob, hist, min_count = 2)
                        entropy_val = tpd_entropy(prob)

                    window_results.append([window, i, j, richness, entropy_val])
                    csvwriter.writerow([window, i, j, richness, entropy_val])

            results_FR[window] = window_results

    return results_FR
        
