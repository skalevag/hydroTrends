"""
Amalie Skålevåg, 20.11.2019
"""
# modules
import HTfunctions as ht
import numpy as np
import datetime as dt
import pandas as pd
from pathlib import Path

## parameters
var = input("Variable: ")
MA = int(input("Moving average (days): "))
folder = "Data/contributingProportion" # location of timeseries .csv files 
# assuming either period 1983-2012 or 1963-2012
# if final catchment list comes from other source, these parameter must be modified accordingly
finalCatchments = ht.openDict("Data/finalSelectionList.pkl") 
region = input("Region [ost,vest]: ") 
startYear = 1983
endYear = 2012
periodIdentifier = "1983-2012"
order = finalCatchments[region][periodIdentifier]

## smoothing and reshaping
arrays = []

for c in order:
    try:
        ts = ht.loadContributionData(c,var,folder=folder)
        smoothed = ht.extractMA(ts,MA,startYear,endYear)
        arrays.append(ht.reshapeTStoArray(smoothed))
    except IndexError:
        print(f"Could not find file for catchment {c}. Filling place in region array with np.nan...")
        arr = np.full_like(arrays[-1],np.nan)
        arrays.append(arr)

regionArray = np.dstack(arrays)
np.save(f"Data/Reshaped/{var}ContRunoff_{region}_{MA}dMA_{startYear}_{endYear}.npy",regionArray)