"""
Docstring...

Daily resolved trend analysis
-----------------------------
Bla bla

References
----------

Resources
---------
"""

import numpy as np
import pandas as pd
import trend
from statsmodels.tsa import stattools
import calendar
from datetime import datetime
from pathlib import Path

# CLASSES
#TODO: finish creating class that is stacked array with methods to perform trend analysis on
class timeSeriesStack:
    """
    This class takes data from a dictionary of time series data, 
    extracts the data for chosen years and applies moving average, 
    and finally stacks the data in a 3D-array, sorted by a specific 
    attribute in a metadata table.
    """
    def __init__(self,data,variable,unit,metadata,sortBy,IDcol,MA,startYear,endYear):
        """
        Initialises timeSeriesStack.

        Parameters
        ----------
        data: dict
            dictionary containing all daily time series as pandas.DataFrames with datetimeIndex
            keys must correspond to contents in "ID" column of metadata
        variable: str
            variable name
        unit: str
            unit of variable to be stacked
        metadata: pandas.DataFrame
            metadata table
            must at MINIMUM contain an "ID" column, 
            in addition to one or more attribute columns by which data can be sorted
        sortBy: str
            header of column in metadata table by which to sort the data
        IDcol: str
        MA: int
        startYear: int
        endYear:int 

        Returns
        -------
        metadata: pd.DataFrame
            sorted metadata table
        
        """
        self.metadata = metadata.sort_values(sortBy)
        self.variable = variable
        self.unit = unit
        self.sortAttributeValues = list(self.metadata[sortBy])
        self.IDs = list(self.metadata[IDcol])
        self.movingAverageDays = MA
        self.sortedBy = sortBy
        self.period = f"{startYear}-{endYear}"

        # make smoothed and stacked array
        arrays = list()
        for c in self.IDs:
            smoothed = extractMA(data[c],MA,startYear,endYear)
            #print(smoothed.index)
            arrays.append(reshapeTStoArray(smoothed))
        self.array = np.dstack(arrays)
        
    def makeClimatology():
        self.climatology = np.nanmean(self.array,axis=1).T
    
    def saveToFile(self,name,DIR="./"):
        """
        Save the array and sorted metadata table to file.
        """
        fileName = f"{name}_{self.variable}_sortedBy{self.sortedBy}_{self.movingAverageDays}MA_{period}"
        self.filenameHead = fileName
        np.save(Path(DIR).joinpath(fileName+"_stackedArray.npy"),self.array)
        self.metadata.to_csv(Path(DIR).joinpath(fileName+"_metadata.csv"))


class trendArray: 
    def __init__(self,tsStack):
        self.tsStack = tsStack
        self.unit = tsStack.unit
        self.IDs = tsStack.IDs
        self.sortedBy = tsStack.sortedBy
 
    def mag(self,method="theil-sen",change="abs",applyPrewhitening=True):
        """
        Calculates the trend magnitudes from the predetermined parameters.

        Parameters
        ----------
        method: str
            "theil-sen"
        change: str
            "abs"   absolute change
            "rel"   relative change
        """
        if method=="theil-sen":
            magnitudes = trendMagnitude(self.tsStack.array,applyPrewhitening=applyPrewhitening)
        else:
            raise Exception("Accepted methods: ['theil-sen']")
        
        if change == "rel":
            self.magnitudes = (magnitudes/tsStack.climatology)*100
            self.trendUnit = "% / yr"
        elif change == "abs":
            self.magnitudes = magnitudes
            self.trendUnit = tsStack.unit + " / yr"

    def sign(self,method="mann-kendall",alpha=0.1,applyPrewhitening=True):
        self.significanceLevel = alpha
        if method=="mann-kendall":
            self.significance = trendSignificance(self.tsStack.array,alpha,applyPrewhitening=applyPrewhitening)
        else:
            raise Exception("Accepted methods: ['mann-kendall']")

    def fieldSign(self,alpha):
        #TODO: finish this
        self.fieldSignificance = "method is not yet implemented" #1D array
    
    def saveToFile(self,DIR="./",name=tsStack.filenameHead):
        """
        Save the trend arrays to file.
        """
        np.save(Path(DIR).joinpath(name + "_trendMagnitudes.npy"),self.magnitudes)
        np.save(Path(DIR).joinpath(name + "_trendSignificance.npy"),self.significance)

        

## FUNCTIONS
# calculating moving averages
def extractMA(timeseries, interval, startYear, endYear, removeFeb29 = True):
    """
    Exstracts moving average timeseries in given timeperiod 
    
    Parameters
    ----------
    timeseries: 
        a pandas timeseries, with datetime index
    interval: int
        number of days in window
    startYear: int
        start of timeseries
    endYear: int
        end of timeseries, 
        i.e. last year to be INCLUDED in the extracted timeseries
        
    Returns
    -------
    timeseries for given years
    """
    #start = datetime(startYear,1,1)
    #end = datetime(endYear+1,1,1)
    
    years = np.arange(startYear,endYear+1)
    
    if removeFeb29:
        # removing feb 29 values in leap years
        # important for reshaping to arrays later
        for year in years:
            if calendar.isleap(year):
                d = datetime(year,2,29)
                d = d.strftime(format="%Y-%m-%d")
                # method 1
                try:
                    mask = ~(timeseries.index==d)
                    timeseries = timeseries[mask]
                # method 2
                except:
                    None
                else:
                    try:
                        d = timeseries.loc[d].index
                        timeseries.drop(d,inplace=True)
                    except:
                        None
                
    
    series = timeseries.rolling(interval,center=True).mean()
    return series[f"{startYear}":f"{endYear}"]


# reshaping to array
def reshapeTStoArray(data):
    """
    Reshapes moving average smoothed data of one catchment from timeseries to array.
    
    OBS! 
    - Timeseries has already been extracted for a given period.
    - Feb 29 has been removed
    
    Parameters
    ----------
    data: pandas.DataFrame
        timeseries from a single catchment
    
    Returns
    -------
    numpy.array
        array of shape (doy,year) with the catchments ordered by altitude
    """
    
    dataYears = np.unique(data.index.year)

    # filling array
    array = []
    for y in dataYears:
        array.append(np.array(data[f"{y}"].iloc[:,0]))
        #print(np.array(data[f"{y}"].iloc[:,0]).shape)
    return np.stack(array,axis=1)

# daily trend analysis
def autocorrTest(ts,alpha=0.05):
    """
    Ljung-Box test for significant autocorrelation in a time series.
    """
    p = stattools.acf(ts,qstat=True,nlags=1)[2]
    p = p[0]
    sign = p < alpha
    return sign

def prewhiten(ts):
    """
    Pre-whitening procedure of a time series.
    
    After Wang&Swail, 2001:
    https://doi.org/10.1175/1520-0442(2001)014%3C2204:COEWHI%3E2.0.CO;2
    """
    r = stattools.acf(ts,nlags=1)[1]
    pw = ts.copy()
    for i in range(ts.shape[0]-1):
        if i > 0:
            pw[i] = (ts[i] - r*ts[i-1])/(1 - r)
    return pw

def trendMagnitude(array,applyPrewhitening=True):
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
                if applyPrewhitening & autocorrTest(ts):
                    ts = prewhiten(ts)
                out[day] = trend.sen_slope(ts)
        output.append(out)
    return np.array(output)

def trendSignificance(array,alpha,applyPrewhitening=True):
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
                if applyPrewhitening & autocorrTest(ts):
                    ts = prewhiten(ts)
                p = trend.mann_kendall(ts) #calculate p-value
                if p < alpha: #calculate trend magnitude if significant trend is detected
                    out[day] = 1
        output.append(out)
    return np.array(output)

#TODO: add CI bootstrap method