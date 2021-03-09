"""
Amalie Skålevåg, 20.11.2019
"""
# modules
from HTfunctions import extractMA,reshapeTStoArray, readRain, readRunoff,readSeNorge,readSnow,readTemp
import numpy as np
import datetime as dt
import pandas as pd
from pathlib import Path
import sys
import importlib

## parameters
# import from setup script
from MAextractReshape_input import var,MA,region,startYear,endYear,catchments,folder,destinationFolder,function

print("The following parameters were imported:")
print("Variable:",var)
print("Moving average:",MA)
print("Region:",region)
print("From",startYear,"to",endYear)
print(len(catchments),"catchments")
print("Location of timeseries .csv files:", folder) # location of timeseries .csv files 
print("\nWorking...")

## smoothing and reshaping
arrays = []

for c in catchments:
    print(c)
    try:
        ts = function(c,folder=folder)
        smoothed = extractMA(ts,MA,startYear,endYear)
        #print(smoothed.index)
        arrays.append(reshapeTStoArray(smoothed))
    except IndexError:
        print(f"Could not find file for catchment {c}. Filling place in region array with np.nan...")
        arr = np.full_like(arrays[-1],np.nan)
        arrays.append(arr)

regionArray = np.dstack(arrays)
np.save(f"{destinationFolder}/{var}_{region}_{MA}dMA_{startYear}_{endYear}.npy",regionArray)

print("Reshaped array saved in",f"{destinationFolder}/{var}_{region}_{MA}dMA_{startYear}_{endYear}.npy")