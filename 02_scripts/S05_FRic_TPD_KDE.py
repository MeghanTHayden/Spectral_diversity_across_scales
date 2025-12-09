#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute FRic using Emiliano Cimoli's KDE/TPD workflow on PCA components
derived from NEON mosaics stored in S3.

Flow:
1. Pull mosaic from S3 for a given site & plot.
2. Run PCA on reflectance.
3. Save PCA stack as GeoTIFF.
4. Use diversity / data_imp.xarray_imp / div_algorithms from Emiliano's repo
   to run KDE-based FRic at a chosen window size.
"""

import os
import re
import argparse
import sys
import time

import boto3
import numpy as np
import rioxarray as rxr
import xarray as xr
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
import multiprocessing as mp
import pandas as pd

# ---------------------------------------------------------------------
# 1. Make sure Emiliano's repo is importable
# ---------------------------------------------------------------------
DIVERSITY_REPO = "/home/ec2-user/Spectral_diversity_across_scales/Functional-Trait-Diversity-FTD-"  # <-- CHANGE if different
if DIVERSITY_REPO not in sys.path:
    sys.path.append(DIVERSITY_REPO)

# These imports correspond to the script you pasted from the repo
from xarray_imp import load_traits_rioxr
from data_manager import normalise_traits_xr
from div_algorithms import (
    generate_ndcell_xr,
    kernel_density_pix,
    kde_fric_pix,
    kde_fdiv_pix,
)
# (GTiff_stack is not strictly needed if we manually create a stacked GeoTIFF)

# ---------------------------------------------------------------------
# 2. Arguments and global settings
# ---------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Compute KDE-based FRic from PCA using Emiliano's workflow."
)
parser.add_argument("--SITECODE", type=str, required=True, help="Site code, e.g. BART")
parser.add_argument("--window_size", type=int, required=True,
                    help="Kernel/window size in pixels (e.g., 31)")
parser.add_argument("--plot_id", type=str, default=None,
                    help="Optional single plot ID; if omitted, loop all plots")
args = parser.parse_args()

SITECODE = args.SITECODE
KERNEL_SIZE = args.window_size  # e.g., 31 (must be odd)
KERNEL_SIZES = [121,241,481,961, 1201,1501,2001]
SINGLE_PLOT = args.plot_id

# Paths
DATA_DIR = "/home/ec2-user/Spectral_diversity_across_scales/01_data/02_processed"
OUT_DIR = "/home/ec2-user/Spectral_diversity_across_scales/03_output"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# S3 setup
bucket_name = "bioscape.gra"
s3 = boto3.client("s3")

# PCA settings
N_PC = 3

# KDE/TPD settings (mirroring Emiliano's defaults as much as possible)
SCALING_METHOD = "normalise"
N_CELLS = 10          # number of cells per dimension in trait space
NORM_MIN = 0.0
NORM_MAX = 1.0
BANDWIDTH = 0.05      # KDE bandwidth
LAT_STRIDE = 40       # stride in raster pixels
LON_STRIDE = 40
PERCENTILES = (0.5, 99.5)
PROB_THRESH = 2.0     # probability threshold for FRic
KERNEL_NAN_THRESH = 70  # % non-NaN above which kernel is skipped


# ---------------------------------------------------------------------
# 3. Helper: list mosaic plots on S3
# ---------------------------------------------------------------------

def list_plots_for_site(sitecode: str):
    dirpath = f"{sitecode}_flightlines/"
    resp = s3.list_objects_v2(Bucket=bucket_name, Prefix=dirpath)
    contents = resp.get("Contents", [])
    mosaics = [
        obj["Key"]
        for obj in contents
        if obj["Key"].endswith(".tif") and "Mosaic_" in obj["Key"]
    ]

    mosaic_names = set()
    for tif in mosaics:
        match = re.search(r"(.*?)_flightlines/Mosaic_(.*?)_(.*?).tif", tif)
        if match:
            mosaic_name = match.group(3)
            mosaic_names.add(mosaic_name)
    return sorted(list(mosaic_names))


# ---------------------------------------------------------------------
# 4. Main processing for one plot
# ---------------------------------------------------------------------

def process_plot(plot_id: str):
    print(f"\n==== Processing {SITECODE} plot {plot_id} ====")

    # --- 4.1 Download mosaic from S3 ---
    s3_key = f"{SITECODE}_flightlines/Mosaic_{SITECODE}_{plot_id}.tif"
    local_mosaic = os.path.join(DATA_DIR, "mosaic.tif")
    print(f"Downloading s3://{bucket_name}/{s3_key} -> {local_mosaic}")
    s3.download_file(bucket_name, s3_key, local_mosaic)

    # --- 4.2 Open raster and run PCA ---
    raster = rxr.open_rasterio(local_mosaic, masked=True)  # (bands, y, x)
    print("Raster shape:", raster.shape)
    bands, ny, nx = raster.shape

    veg_np = raster.values  # (bands, ny, nx)
    X = veg_np.reshape(bands, ny * nx).T  # (n_pixels, bands)
    X = X.astype("float32")
    X[X <= 0] = np.nan

    print("NaN proportion:", np.isnan(X).mean())

    # Scale reflectance roughly to [0,1]
    X /= 10000.0

    # Impute + standardize
    imputer = SimpleImputer(missing_values=np.nan, strategy="median")
    X_imp = imputer.fit_transform(X)

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_imp)

    # PCA
    print("Fitting PCA...")
    pca = PCA(n_components=N_PC)
    pca_scores = pca.fit_transform(X_scaled)  # (n_pixels, N_PC)
    print("Explained variance:", pca.explained_variance_ratio_)

    # Reshape PC scores back to raster shape: (traits, y, x)
    pca_stack = pca_scores.T.reshape(N_PC, ny, nx)

    # --- 4.3 Wrap PCA stack as xarray DataArray & write GeoTIFF ---
    #    (This is simpler than trying to rewire all I/O in Emiliano's code.)
    trait_names = [f"PC{k}" for k in range(1, N_PC + 1)]
    pca_da = xr.DataArray(
        pca_stack,
        dims=("traits", "y", "x"),
        coords={
            "traits": trait_names,
            "y": raster.y.values,
            "x": raster.x.values,
        },
        name="traits",
    )
    pca_da = pca_da.rio.write_crs(raster.rio.crs)

    pca_tif = os.path.join(
        OUT_DIR, f"{SITECODE}_{plot_id}_PCA_stack_{N_PC}pc.tif"
    )
    print(f"Writing PCA stack to {pca_tif}")
    pca_da.rio.to_raster(pca_tif)

    # --- 4.4 Use Emiliano's pipeline from here on ---

    # Load PCA stack as traits_xr
    traits_xr = load_traits_rioxr(pca_tif, traits=trait_names)

    # Normalise traits (0–1) based on percentiles
    traits_xr_norm = xr.apply_ufunc(
        normalise_traits_xr,
        traits_xr.groupby("traits"),
        dask="parallelized",
        keep_attrs=True,
        kwargs={"percentiles": PERCENTILES},
    )

    # Write normalized stack (for reloading as dask)
    pca_norm_tif = os.path.join(
        OUT_DIR, f"{SITECODE}_{plot_id}_PCA_stack_{N_PC}pc_norm.tif"
    )
    traits_xr_norm.rio.to_raster(pca_norm_tif)

    # Re-load as chunked xarray
    traits_xr = load_traits_rioxr(
        pca_norm_tif, traits=trait_names, band=N_PC, x=300, y=300
    )
    traits_xr = traits_xr.chunk({"traits": len(trait_names)})

    # Generate voxel space (cells_xr)
    cells_xr = generate_ndcell_xr(
        N_CELLS, NORM_MIN, NORM_MAX, ndim=traits_xr.shape[0], trait_names=trait_names
    )

    # --- 4.5 Rolling window and KDE, single kernel size ---
   results = []
   try:
        # -------------------------------------------------------------
        # Loop over kernel sizes: KDE, FRic/FDiv, median summary
        # -------------------------------------------------------------
        for KERNEL_SIZE in KERNEL_SIZES:
            print(f"---- Running KDE/TPD with KERNEL_SIZE = {KERNEL_SIZE} ----")

            # Rolling window for this kernel size
            rolling_window = traits_xr.rolling(
                latitude=KERNEL_SIZE,
                longitude=KERNEL_SIZE,
                center={"latitude": True, "longitude": True},
                min_periods=1,
            )

            window_construct = rolling_window.construct(
                latitude="x_window",
                longitude="y_window",
                stride={"latitude": LAT_STRIDE, "longitude": LON_STRIDE},
            )

            kernels_xr = window_construct.stack(
                gridcell=("latitude", "longitude"), window=["x_window", "y_window"]
            ).transpose("gridcell", "window", "traits")

            # Drop kernels that are mostly NaN
            threshold = (KERNEL_NAN_THRESH * (KERNEL_SIZE ** 2)) / 100.0
            kernels_xr = kernels_xr.dropna(
                "gridcell", how="all", thresh=threshold
            )

            # Save gridcell coords for later
            sav_gridcell_dim = kernels_xr.indexes["gridcell"]
            print("Please ignore the warning about coords being overwritten")
            kernels_xr.coords["gridcell"] = np.linspace(
                0, kernels_xr.shape[0], kernels_xr.shape[0]
            )

            # Fill remaining NaNs with mean within each kernel
            kernels_xr = kernels_xr.fillna(kernels_xr.mean(dim="gridcell"))

            kernel_groups = kernels_xr.groupby("gridcell")

            # Kernel density per kernel
            t0 = time.perf_counter()
            kernel_density = xr.apply_ufunc(
                kernel_density_pix,
                kernel_groups,
                cells_xr,
                input_core_dims=[["window", "traits"], ["nd_cell", "traits"]],
                output_core_dims=[["nd_cell"]],
                exclude_dims=set(("window",)),
                join="exact",
                dask="allowed",
                keep_attrs=True,
                kwargs={"bandwidth": BANDWIDTH},
                vectorize=False,
            )
            t1 = time.perf_counter()
            print(f"KDE computed in {t1 - t0:.1f} s for kernel {KERNEL_SIZE}")

            # Restore real gridcell coords
            kernel_density = kernel_density.assign_coords(
                gridcell=sav_gridcell_dim.set_names(["latitude", "longitude"])
            )

            # --- FRic ---
            kde_fric_uf = xr.apply_ufunc(
                kde_fric_pix,
                kernel_density.groupby("gridcell"),
                input_core_dims=[["nd_cell", "gridcell"]],
                exclude_dims=set(("nd_cell",)),
                join="exact",
                dask="allowed",
                keep_attrs=True,
                kwargs={"prob_thresh": PROB_THRESH, "norm": 1},
                vectorize=False,
            )

            kde_fric_map = kde_fric_uf.unstack("gridcell")

            # --- FDiv ---
            kde_fdiv_uf = xr.apply_ufunc(
                kde_fdiv_pix,
                kernel_density.groupby("gridcell"),
                cells_xr,
                input_core_dims=[["gridcell", "nd_cell"], ["nd_cell", "traits"]],
                exclude_dims=set(("nd_cell",)),
                join="exact",
                dask="allowed",
                keep_attrs=True,
                kwargs={
                    "ndim": kernels_xr.shape[2],
                    "n_cells": N_CELLS,
                    "prob_thresh": PROB_THRESH,
                },
                vectorize=False,
            )
            kde_fdiv_map = kde_fdiv_uf.unstack("gridcell")

            # --- Build NaN mask aligned with this KDE output ---
            mask_this = mask.assign_coords(
                {"longitude": kde_fric_map.coords["longitude"].values}
            )
            mask_this = mask_this.assign_coords(
                {"latitude": kde_fric_map.coords["latitude"].values}
            )
            mask_this = mask_this.T
            mask_this = xr.DataArray(
                np.rot90(mask_this.data), dims=("latitude", "longitude")
            )
            nan_mask = mask_this.where(mask_this != 0, 1).where(mask_this == 0, 0)
            nan_mask = kde_fric_map.copy(deep=True, data=nan_mask)

            # Apply mask and compute medians (no rasters written)
            masked_fric = nan_mask * kde_fric_map
            masked_fric = masked_fric.where(masked_fric != 0, np.nan)
            med_fric = float(masked_fric.median(skipna=True))

            masked_fdiv = nan_mask * kde_fdiv_map
            masked_fdiv = masked_fdiv.where(masked_fdiv != 0, np.nan)
            med_fdiv = float(masked_fdiv.median(skipna=True))

            print(
                f"Kernel {KERNEL_SIZE}: median FRic = {med_fric:.4f}, "
                f"median FDiv = {med_fdiv:.4f}"
            )

            results.append(
                {
                    "site": SITECODE,
                    "plot_id": plot_id,
                    "kernel_size": KERNEL_SIZE,
                    "n_pc": N_PC,
                    "median_fric": med_fric,
                    "median_fdiv": med_fdiv,
                }
            )

        # After all kernel sizes: write summary CSV for this plot
        out_csv = os.path.join(
            OUT_DIR,
            f"{SITECODE}_{plot_id}_TPD_medians_{N_PC}pc.csv",
        )
        df = pd.DataFrame(results)
        df.to_csv(out_csv, index=False)
        print(f"Saved median FRic/FDiv summary to {out_csv}")

    finally:
        # Clean up big arrays
        try:
            os.remove(local_mosaic)
        except OSError:
            pass

    print(f"Done with {SITECODE} plot {plot_id}.")


# ---------------------------------------------------------------------
# 5. Run over plots
# ---------------------------------------------------------------------

# For one plot
#if __name__ == "__main__":
#    if SINGLE_PLOT is not None:
#        plots = [SINGLE_PLOT]
#    else:
#        plots = list_plots_for_site(SITECODE)

#    print(f"Plots found for {SITECODE}: {plots}")

#    for pid in plots:
#        process_plot(pid)

# With multiprocessing
N_PROCESSES = mp.cpu_count() - 1

if __name__ == "__main__":
    if SINGLE_PLOT is not None:
        plots = [SINGLE_PLOT]
    else:
        plots = list_plots_for_site(SITECODE)

    print(f"Plots found for {SITECODE}: {plots}")
    print(f"Using {N_PROCESSES} processes")

    if len(plots) == 1 or N_PROCESSES == 1:
        # Fall back to serial for debugging / single plot
        for pid in plots:
            process_plot(pid)
    else:
        with mp.Pool(processes=N_PROCESSES) as pool:
            pool.map(process_plot, plots)
