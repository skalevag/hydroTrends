"""
DAILY TREND ANALYSIS
Amalie Skålevåg, 29.11.2019
###########################################################################################################################
Daily trend analysis approach after Kormann, et al. 2014; 2015

Running daily trend analysis for hydroTrends project. 
Using already existing reshaped region arrays containing smoothed data for all catchments in a region, ordered by altitude.
Input array must be a .npy file in the shape (doy,year,catchment).

Trend array is saved to Results/Daily unless another output directory is specified in the command line.
###########################################################################################################################
"""
print(__doc__)

## IMPORTING MODULES AND FUNCTIONS
from HTfunctions import autocorrTest,prewhiten 
import numpy as np
import trend
import sys

## PARAMETERS AND SETUP
inFile = sys.argv[1]
filename = inFile.split("/")[-1]
name = filename.split(".")[0]
print("Analysing the file",filename)

out_dir = "Results/Daily"
try: 
    print(sys.argv[2], "is set as output directory.")
    out_dir = sys.argv[2]
except IndexError:
    print(out_dir, "is set as output directory.")

array = np.load(inFile)
alpha = 0.1

## ANALYSIS
def trendMagnitude(array):
    """
        Calculates the trend magnitude for each doy time series.
        
        Parameters
        ----------
        array: numpy.array
        array of shape: (doy,year,catchment) containing data to be analysed
        
        Returns
        -------
        numpy.array
        array of trend magnitude, shape: (catchments,doy)
        """
    output = []
    for c in range(array.shape[2]):
        arr = array[:,:,c] # slicing array by catchment
        if (arr==-99).all():
            out = np.full(arr.shape[0],-99)
        else:
            out = np.full(arr.shape[0],np.nan) # create empty array
            for day in range(arr.shape[0]):
                ts = arr[day,:]
                if autocorrTest(ts):
                    ts = prewhiten(ts)
                out[day] = trend.sen_slope(ts)
        output.append(out)
    return np.array(output)

def trendSignificance(array,alpha):
    """
        Calculates the trend significance at specified significance level alpha for each doy time series.
        
        Parameters
        ----------
        array: numpy.array
        array of shape: (doy,year,catchment) containing data to be analysed
        
        Returns
        -------
        numpy.array
        array of trend significance, shape: (catchments,doy)
        """
    output = []
    for c in range(array.shape[2]):
        arr = array[:,:,c] # slicing array by catchment
        if (arr==-99).all():
            out = np.full(arr.shape[0],-99)
        else:
            out = np.full(arr.shape[0],0) # create empty array
            for day in range(arr.shape[0]):
                ts = arr[day,:]
                if autocorrTest(ts):
                    ts = prewhiten(ts)
                p = trend.mann_kendall(ts) #calculate p-value
                if p < alpha: #calculate trend magnitude if significant trend is detected
                    out[day] = 1
        output.append(out)
    return np.array(output)

print("\nStarting analysis...")
trendMag = trendMagnitude(array)
trendSign = trendSignificance(array,alpha)
np.save(f"{out_dir}/trendMagnitudes_{name}.npy", trendMag)
np.save(f"{out_dir}/trendSignificance_{name}.npy", trendSign)

print("\nDaily trend analysis complete.")
print(f"Array containing results can be found in {out_dir} directory.")