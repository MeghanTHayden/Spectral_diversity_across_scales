#!/usr/bin/env python3i
# -*- coding: utf-8 -*-
"""
Script that loops through individual files to correct, clip, mosaic and compute FRic & FDiv for plots within NEON site.

User must define parameters that feed into calling each script. 

Requires:
- correction coefficients
- shapefiles (plot boundaries)


Author: M. Hayden
Last Updated: 1/23/24
"""

# Load required libraries
import os

# Define variables
SITECODE = 'SRER'
DOMAIN_ID = 'D14'
ID_NO = '3'
YEAR = '201909'
DATE = '20190903'
DATE_ID = '2019090314'
EPSG = '32612'
NDVI_THRESHOLD = 0.35
NIR_THRESHOLD = 0.2


# Workflow to get Functional Richness and Functional Divergence for set of NEON plots (within a site)

# 1. Topo/BRDF corrections
script_string = (f"python 02_scripts/S02_TopoBRDF_Corrections.py --SITECODE {SITECODE} " 
                f"--DOMAIN {DOMAIN_ID} " 
                f"--ID_NO {ID_NO} "  
                f"--DATE {DATE} " 
                f"--DATE_ID {DATE_ID} " 
                f"--EPSG {EPSG} " 
                f"--NDVI {NDVI_THRESHOLD} "
                f"--NIR {NIR_THRESHOLD} ")
os.system(script_string)
print("Corrections complete.")

# 2. Clip to regions of interest
script_string = (f"python 02_scripts/S03_Clip_Corrected.py --SITECODE {SITECODE} " 
                f"--YEAR {YEAR}") 
os.system(script_string)
print("Flightlines clipped.")

# 3. Mosaic flightlines
script_string = (f"python 02_scripts/S04_Mosaic_Clipped_Raster.py --SITECODE {SITECODE} " 
                f"--EPSG {EPSG}")
os.system(script_string)
print("Flightlines mosaicked.")

# 4. Compute Functional Richness
script_string = (f"python 02_scripts/S05_Compute_FRic_FDiv.py --SITECODE {SITECODE}") 
os.system(script_string)
print("FRic & FDiv computed for all mosaics.")

