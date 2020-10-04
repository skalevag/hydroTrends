### Functions used in handling and importing data
import datetime
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import trend
from statsmodels.tsa import stattools
import calendar
#import geopandas as gpd
import descartes
from shapely.geometry import Point,Polygon
import itertools
import matplotlib.pyplot as plt

# Saving and loading functions using pickle

def saveDict(dictionary,filename):
    """
    Saves dictionary to pickle file in working directory.
    
    Parameters
    ----------
    dictionary: dict
    filename: str
        filename without .pkl ending
    
    Returns
    -------
    Nothing
    """
    f = open(f"{filename}.pkl","wb")
    pickle.dump(dictionary,f)
    f.close()

def openDict(filename):
    """
    Opens dictionary from pickle file in working directory, unless other folder is specified.
    
    Parameters
    ----------
    filename: str
        filename with .pkl ending
    
    Returns
    -------
    dictionary
    """
    pickle_in = open(f"{filename}","rb")
    loadedDict = pickle.load(pickle_in)
    return loadedDict

def findFiles(search,directory):
    """
    Finds files in a set directory.
    
    Parameters
    ----------
    search: str
        term used to search for files in directory
    directory: str
        path to directory
    
    Returns
    -------
    list of files
    """
    # set folder
    folder = Path(directory)
    
    # make list of strings
    files = []
    for item in sorted(folder.glob(search)):
        files.append(str(item))
    
    return files

# from/to stnr to/from snumber
def stnr_to_snumber(regine,main):
    """
    Makes an snumber from stnr in form regine.main
    
    Parameters:
    -----------
    regine: int
    main: int
    
    Returns:
    snumber: str
    """
    main = "{:>03}".format(main)
    sn = str(regine)+"00"+str(main)
    
    return sn

def snumber_to_stnr(snumber):
    """
    Makes an stnr in form (regine,main) from snumber
    
    Parameters:
    -----------
    snumber: str
    
    Returns:
    tuple of integers (regine,main)
    """
    snumber = str(snumber)
    main = int(snumber[-3:])
    regine = int(snumber[:-5])
    
    return regine,main

# importing SeNorge and runoff data
def readRunoff(catchmentNo,folder='Data/runoff'):
    """
    Reads streamflow data in 'Data/runoff' folder for a specific catchment number.

    Parameters
    ----------
    catchmentNo: str
        snumber of catchment
    folder: str
    
    Returns
    -------
    pandas.DataFrame
    """
    # reading data
    runoff = pd.read_csv(f"{folder}/{catchmentNo}.txt",
                         header=None, sep=" ", names=["date","stream"],usecols=[0,1],index_col=0)
    # converting indices to datetime objects
    runoff.index = pd.to_datetime(runoff.index)
    # replacing invalid values with nan
    runoff = runoff.replace(-9999.,np.nan)

    return runoff

def readSnow(snumber,folder="Data/seNorge"):
    """
    Reads snowmelt data in 'Data/seNorge' folder for catchments specified by regine and main number.

    Parameters
    ----------
    regine: str
    main: str
    folder: str

    Returns
    -------
    pandas.DataFrame
    """
    # get regine and main from snumber
    snumber = str(snumber)
    regine, main = snumber_to_stnr(snumber)
    # create filepath
    file = f"{folder}/{regine}.{main}/{regine}.{main}_SeNorge_qsw_1959_2014.dta"
    # read file
    snow = pd.read_table(file,delim_whitespace=True,header=None,names=["year","month","day","qsw"])
    # list of dates
    index = list(np.arange(len(snow)))
    for i in range(len(snow)):
        index[i] = f"{snow.year.iloc[i]}-{snow.month.iloc[i]}-{snow.day.iloc[i]}"
    # index as timestamp objects
    snow.index = pd.to_datetime(index)
    # removing unnecessary columns
    snow = snow.drop(["year","month","day"],axis = 1)

    return snow

def readSeNorge(snumber,folder="Data/seNorge"):
    """
    Reads temperature and rainfall data in 'Data/seNorge' folder for catchments specified by regine and main number.

    Parameters
    ----------
    regine: str
    main: str
    folder: str

    Returns
    -------
    pandas.DataFrame
    """
    # get regine and main from snumber
    snumber = str(snumber)
    regine, main = snumber_to_stnr(snumber)
    # create filepath
    file = f"{folder}/{regine}.{main}/{regine}.{main}_SeNorge_rr_tm_1959_2014.dta"
    # read file
    data = pd.read_table(file,delim_whitespace=True,header=None,names=["year","month","day","rain","temp"])
    # list of dates
    index = list(np.arange(len(data)))
    for i in range(len(data)):
        index[i] = f"{data.year[i]}-{data.month[i]}-{data.day[i]}"
    # index as timestamp objects
    data.index = pd.to_datetime(index)
    # removing unnecessary columns
    data = data.drop(["year","month","day"],axis = 1)
    return data

def readRain(snumber,folder="Data/seNorge"):
    """
    Reads temperature and rainfall data in 'Data/seNorge' folder for catchments specified by regine and main number.

    Parameters
    ----------
    regine: str
    main: str
    folder: str

    Returns
    -------
    pandas.DataFrame
    """
    # get regine and main from snumber
    snumber = str(snumber)
    regine, main = snumber_to_stnr(snumber)
    # create filepath
    file = f"{folder}/{regine}.{main}/{regine}.{main}_SeNorge_rr_tm_1959_2014.dta"
    # read file
    data = pd.read_table(file,delim_whitespace=True,header=None,names=["year","month","day","rain","temp"])
    # list of dates
    index = list(np.arange(len(data)))
    for i in range(len(data)):
        index[i] = f"{data.year[i]}-{data.month[i]}-{data.day[i]}"
    # index as timestamp objects
    data.index = pd.to_datetime(index)
    # removing unnecessary columns
    data = data.drop(["year","month","day","temp"],axis = 1)
    return data

def readTemp(snumber,folder="Data/seNorge"):
    """
    Reads temperature and rainfall data in 'Data/seNorge' folder for catchments specified by regine and main number.

    Parameters
    ----------
    regine: str
    main: str
    folder: str

    Returns
    -------
    pandas.DataFrame
    """
    # get regine and main from snumber
    snumber = str(snumber)
    regine, main = snumber_to_stnr(snumber)
    # create filepath
    file = f"{folder}/{regine}.{main}/{regine}.{main}_SeNorge_rr_tm_1959_2014.dta"
    # read file
    data = pd.read_table(file,delim_whitespace=True,header=None,names=["year","month","day","rain","temp"])
    # list of dates
    index = list(np.arange(len(data)))
    for i in range(len(data)):
        index[i] = f"{data.year[i]}-{data.month[i]}-{data.day[i]}"
    # index as timestamp objects
    data.index = pd.to_datetime(index)
    # removing unnecessary columns
    data = data.drop(["year","month","day","rain"],axis = 1)
    return data

def organiseData(region):
    """
    Organises all data for a region into a dictionary.
    
    Parameters
    -----------
    region: pandas.DataFrame
        dataframe containing metadata on all catchments in region
    
    Returns
    -----------
    dictionary with all variables
    """
    # create dictionary and adds station order by elevation
    d = {"order":list(region.snumber),
         "data":{},
         "metadata":region}
    # add all catchments as keys
    for i in range(region.shape[0]):
        d["data"][region.snumber.iloc[i]] = {}
    
    for c in d["order"]:
        # runoff data for each cathcment
        d["data"][c]["stream"] = readRunoff(c)
        # snowmelt    
        regine = int(region[region.snumber == c].regine)
        main = int(region[region.snumber == c].main)
        d["data"][c]["snow"] = readSnow(regine,main)
        # precipitation and temperature
        pt = readSeNorge(regine,main)
        d["data"][c]["temp"] = pt.temp
        d["data"][c]["rain"] = pt.rain
        
    # return the finished dictionary
    return d

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

# loading the contribution to runoff data from .csv files
def loadContributionData(c,var,folder = "Data/contributingProportion"):
    if var == "rainfall" or var == "rain" or var == "rf":
        file = findFiles(f"*{c}*rf*.csv",folder)[0]
    elif var == "snowmelt" or var == "snow" or var == "sm":
        file = findFiles(f"*{c}*sm*.csv",folder)[0]
    
    df = pd.read_csv(file,index_col="date")
    df.index = pd.to_datetime(df.index)
    
    return df

# reshaping time series to array
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

# trend analysis functions
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

def trendMagnitude(array,alpha=0.1):
    """
        Calculates the trend magnitude for each doy if a significant trend is detected.
        
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
                p = trend.mann_kendall(ts) #calculate p-value
                if p < alpha: #calculate trend magnitude if significant trend is detected
                    out[day] = trend.sen_slope(ts)
        output.append(out)
    return np.array(output)

    
def splitBySeason(array,season,firstDayOfSeasons = (60,152,244,335)):
    """
    Slices 2D array with shape (catchments,doy) or a 1D with shape (doy,) by a specified season.
    
    Parameters
    ----------
    array: np.array
        shape: (catchments,doy)
    season: str
        "spring","sp"
        "summer","su"
        "autumn","fall","au"
        "winter","wi"
    firstDayOfSeasons: tuple (springday,summerday,autumnday,winterday)
        springday: first day of spring, default 01.03.
        summerday: first day of summer, default 01.06.
        autumnday: first day of autumn, default 01.09. (also first day of norwegian hydrological year)
        winterday: first day of winter, default 01.12.
    
    Returns
    -------
    arr: np.array
        array sliced by season
    """
    
    springday,summerday,autumnday,winterday = firstDayOfSeasons
    
    # 2D arrays
    if len(array.shape)==2:
        if (season=="spring" or season=="sp"):
            start = springday-1
            end = summerday-1
            arr = array[:,start:end]
        elif (season=="summer" or season=="su"):
            start = summerday-1
            end = autumnday-1
            arr = array[:,start:end]
        elif (season=="autumn" or season=="fall" or season=="au"):
            start = autumnday-1
            end = winterday-1
            arr = array[:,start:end]
        elif (season=="winter" or season=="wi"):
            a = array[:,winterday-1:]
            b = array[:,:springday-1]
            arr = np.hstack([a,b])
        else:
            print("Please input correct season identifier.")
    
    # 1D arrays
    elif len(array.shape)==1:
        if (season=="spring" or season=="sp"):
            start = springday
            end = summerday
            arr = array[start:end]
        elif (season=="summer" or season=="su"):
            start = summerday
            end = autumnday
            arr = array[start:end]
        elif (season=="autumn" or season=="fall" or season=="au"):
            start = autumnday
            end = winterday
            arr = array[start:end]
        elif (season=="winter" or season=="wi"):
            a = array[winterday:]
            b = array[:springday]
            arr = np.hstack([a,b])
        else:
            print("Please input correct season identifier.")
    
    # error if array is more than 2D
    else:
        print("Array must be 1D or 2D.")
    
    return arr

def getStations(df,qual = "all"):
    """
    Extracts station locations for a given dataframe.
    Optional to filter by r-squared using "qual" parameter.
    
    Parameters
    ----------
    df: pandas.DataFrame
    qual: int,str 
        default: "all"
            all stations in df are returned
        int {0,1,2,3,4}
            if integer between 0 and 4, the stations are filtered according to r-squared 
            where 0 is worst and 4 best quality
    
    Returns
    -------
    geopandas.geodataframe.GeoDataFrame
        containing specified catchments
    """
    stations = gpd.read_file("Data/gis/Gauging_Stations.shp")
    
    filters = [(df.rsquared_adj>=0) & (df.rsquared_adj<0.2),
               (df.rsquared_adj>=0.2) & (df.rsquared_adj<0.4),
               (df.rsquared_adj>=0.4) & (df.rsquared_adj<0.6),
               (df.rsquared_adj>=0.6) & (df.rsquared_adj<0.8),
               (df.rsquared_adj>=0.8) & (df.rsquared_adj<=1)]
    
    if qual=="all":
        sel = []
        for sn in df.snumber:
            sel.append(".".join(str(i) for i in snumber_to_stnr(sn)))
    else:
        sel = []
        filt = filters[qual]
        for sn in df[filt].snumber:
            sel.append(".".join(str(i) for i in snumber_to_stnr(sn)))

    mask = stations.STASJON_NR.isin(sel)
    
    return stations[mask]

def plotRegressionQualityLocation(reg1,reg2):
    """
    Plotting catchments colorcoded by quality of regression (R²).

    Parameters
    ----------
    reg1: pandas.DataFrame
    reg2: pandas.DataFrame
    """
    norge = gpd.read_file("Data/gis/Norge.shp")
    regionLines = gpd.read_file("Data/gis/Grense_Avrenningsregioner.shp")
    
    # quality colors to use in map
    cmap = plt.cm.get_cmap("jet_r")

    qcol = []
    for i in np.linspace(0.1,0.9,5):
        qcol.append(cmap(i))

    qlabel = ["$R^2$ < 0.2",
              "$R^2$ = 0.2-0.4",
              "$R^2$ = 0.4-0.6",
              "$R^2$ = 0.6-0.8",
              "$R^2$ > 0.8"]
    
    # plotting and saving map
    fig,ax = plt.subplots(figsize=(7,7))
    norge.plot(ax=ax,color="white", edgecolor="lightgrey")
    regionLines.plot(ax=ax,color="darksalmon",label="Runoff region border")

    for qual in np.arange(5):
        stationSel = getStations(reg1,qual)
        stationSel.plot(ax=ax,marker="o",markersize=60,color=qcol[qual],label=qlabel[qual],alpha=.6)
        stationSel = getStations(reg2,qual)
        stationSel.plot(ax=ax,marker="o",markersize=60,color=qcol[qual],label="",alpha=.6)

    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    plt.setp(ax.get_xticklines(), visible=False)
    plt.setp(ax.get_yticklines(), visible=False)
    plt.setp(ax.spines.values(), visible=False)
    plt.ylim(6450000,7300000)
    plt.xlim(-100000,400000)
    plt.legend(loc="upper left")

def plotCorrMatrix(corr,figsize=(7,7)):
    matfig = plt.figure(figsize=figsize)
    plt.matshow(corr,cmap="coolwarm",vmin=-1,vmax=1,fignum=matfig.number)
    N = corr.shape[0]
    for col in range(N):
        for row in range(N):
            plt.text(row,col,f"{corr.iloc[col,row]:.2f}",ha="center",va="center")
    plt.xticks(range(N),corr.columns,rotation="vertical")
    plt.yticks(range(N),corr.columns)
    plt.ylim(N-0.5,-0.5)
    plt.xlim(-0.5,N-0.5)