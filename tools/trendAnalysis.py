import numpy as np
import trend
from statsmodels.tsa import stattools

def autocorrTest(ts,alpha=0.05):
    """
        Ljung-Box test for significant autocorrelation in a time series.
        """
    p = stattools.acf(ts,qstat=True,nlags=1)[3]
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