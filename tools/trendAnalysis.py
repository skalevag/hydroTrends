import numpy as np
import trend
from statsmodels.tsa import stattools
import calendar
import datetime

#TODO: create class that is stacked array with methods to do trend analysis
#method 1: moving average
#method 2: trend analysis

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
    #start = datetime.datetime(startYear,1,1)
    #end = datetime.datetime(endYear+1,1,1)
    
    years = np.arange(startYear,endYear+1)
    
    if removeFeb29:
        # removing feb 29 values in leap years
        # important for reshaping to arrays later
        for year in years:
            if calendar.isleap(year):
                d = datetime.datetime(year,2,29)
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
