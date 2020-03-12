"""
Calculated fieldsignificance for all arrays Reshaped folder.
"""

import numpy as np
import pandas as pd
import datetime
import pickle
import trend
from pathlib import Path
from HTfunctions import findFiles
import datetime


def resamplingDaily(array):
    """
        Resampling procedure after Burn and Hag Elnur, 2002.
        
        Parameters
        ----------
        array: 3D numpy array in the shape (DOY,years,catchments)
        years: int
        MA: str
        
        Returns
        -------
        dataframe with resampled data
        """
    years = np.arange(0,array.shape[1])
    days = np.arange(0,array.shape[0])
    resampled = np.full_like(array,np.nan)
    for DOY in days:
        for i in range(len(years)):
            # select random year
            year = np.random.choice(years)
            # get values for that year
            resampled[DOY,i,:] = array[DOY,year,:]

    return resampled

def fieldSignDaily(array, alpha = 0.1, q = 90, NS = 600):
    """
    Calculating the field significance after Burn and Hag Elnur, 2002.
    """
    days = np.arange(0,array.shape[0])
    
    significant = []
    print("Iternation number:")
    for i in range(NS):
        if i in np.arange(0,NS,50):
            print(datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S"),i,f"of {NS}")
        resampledArray = resamplingDaily(array)
        sign = np.full(resampledArray.shape[0],np.nan)
        for d in days:
            if np.isfinite(resampledArray[d,:,0]).all():
                s = 0
                for c in range(resampledArray.shape[2]):
                    p = trend.mann_kendall(resampledArray[d,:,c])
                    if p<alpha:
                        s += 1
                # proportion of catchments with signifcant trend
                sign[d] = s/resampledArray.shape[2]
        significant.append(sign)
    
    distribution = np.array(significant)
    
    pcrit = []
    for d in days:
        if np.isfinite(distribution[:,d]).all():
            pcrit.append(np.percentile(distribution[:,d],q))
        else:
            pcrit.append(np.nan)
    pcrit = np.array(pcrit)
    
    percentSign = []
    for d in days:
        s = 0
        for c in range(array.shape[2]):
            p = trend.mann_kendall(array[d,:,c])
            if p<alpha:
                s += 1
        percentSign.append(s/array.shape[2])
    percentSign = np.array(percentSign)
    
    output = {"pcrit":pcrit,"percentSign":percentSign,"fieldSignificant":percentSign>pcrit}
    return pd.DataFrame(output)

variable = input("\n\n-----\nIf selecting files by VARIABLE please enter 'streamflow','rainfall' or 'snowmelt', else press Enter\n")
region = input("\nIf selecting files by REGION please enter shortend name of region,e.g. 'ost' etc, else press Enter\n")
MA = input("\nIf selecting files by MA smoothing please enter shortend window size,e.g. '5', else press Enter\n")
period = input("\nIf selecting files by period please enter number of years,e.g. '1983', else press Enter\n")
NS = input("\nThe default resampling number NS is 600, if you would like to specify a different please enter number below, else press Enter\n")


if variable == "":
    variable="_"
if region == "":
    region="_"
if MA == "":
    MA="d"
if period == "":
    period = None
if NS == "":
    NS = 600
else:
    NS = int(NS)


files = findFiles(f"*{variable}*{region}*{MA}*{period}*.npy","Data/Reshaped")
print("\n---------------------------------")
print(datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S"))
print(f"Analysing {len(files)} files.")
for file in files:
    print(file)
print("---------------------------------\n")

for file in files:
    name = file.split("/")[-1].split(".")[0]
    # checking if file already exists
    exists = findFiles(f"*{name}*{NS}iter*","Results/FS")
    if len(exists)>0:
        print("-----","\nFile already exists in Results/FS directroy","\nField significance not calculated for",file,"\n-----")
        continue
    else:
        print(datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S"),file,"calculating...")
    # opening array file
    array = np.load(file)
    # calculating field significance
    result = fieldSignDaily(array,NS=NS)
    result.to_csv(f"Results/FS/fieldSignificance_{name}_{NS}iter.csv")
    print(datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S"),name,"finished.\n")
