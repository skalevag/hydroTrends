
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams, cycler
import matplotlib
from matplotlib.lines import Line2D
import trend
import itertools
from scipy import stats
import datetime
import pickle
from pathlib import Path
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter


# ## Loading data

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
    Opens dictionary from pickle file in working directory.
    
    Parameters
    ----------
    filename: str
        filename without .pkl ending
    
    Returns
    -------
    dictionary
    """
    pickle_in = open(f"{filename}.pkl","rb")
    loadedDict = pickle.load(pickle_in)
    return loadedDict

def findFS(variable="_",region="_",MA="day",years="year",resultDir="Data/Reshaped"):
    """
    Finds .npy-files for a specific variable and moving average smoothing (MA) in a directory.
    
    Parameters
    ----------
    variable: str
    MA: str
    
    Returns
    -------
    list of files
    """
    # set folder
    folder = Path(resultDir)
    
    # make list of strings
    files = []
    for item in sorted(folder.glob(f"*{variable}*{region}*{MA}*{years}*")):
        files.append(str(item))
    
    return files





# # Plotting daily trends

# ## Functions

# In[5]:


regionLabels = {"ost":"Østlandet",
                "vest":"Vestlandet",
                "nord":"Nordland",
                "finn":"Finnmark",
                "trond":"Trøndelag",
                "sor":"Sørlandet"}
units = {"runoff":"$m^3\ s^{-1}\ yr^{-1}$",
         "spesrunoff":"$m^3\ s^{-1}km^{-2}\ yr^{-1}$",
         "temperature":"$^{\circ}C\ yr^{-1}$",
         "temp":"$^{\circ}C\ yr^{-1}$",
         "rainfall":"$mm\ yr^{-1}$",
         "snowmelt":"$mm\ yr^{-1}$",
         "streamflow":"$m^3\ s^{-1}km^{-2}\ yr^{-1}$"}
v = {"runoff":"streamflow",
       "spesrunoff":"specific streamflow",
       "temperature":"temperature",
       "rainfall":"rainfall",
       "snowmelt":"snowmelt",
       "streamflow":"streamflow"}

def findFiles(variable,MA,years,head="trend",resultDir="Results"):
    """
    Finds .npy-files for a specific variable and moving average smoothing (MA) in a directory.
    
    Parameters
    ----------
    variable: str
    MA: str
    
    Returns
    -------
    list of files
    """
    # set folder
    folder = Path(resultDir)
    
    # make list of strings
    files = []
    for item in sorted(folder.glob(f"*{head}*_{variable}*{MA}*{years}*.npy")):
        files.append(str(item))
    
    return files

def plotTrendMag(array,variable,region,MA,years,filename,
                 display=False,subplotting=False,ax=None,overrideColormap=None,titleOverride=None,overrideAspect=None):
    """
    Plots a single trend magnitude array.
    """
    
    masked_array = np.ma.masked_where(array == -99, array)
    extent = np.array([np.mean(np.nanmax(masked_array,axis=1)),np.mean(np.nanmin(masked_array,axis=1))]).max()
    
    if variable=="temp" or variable=="temperature":
        cmap=plt.cm.get_cmap("coolwarm")
    else:
        cmap=plt.cm.get_cmap("coolwarm_r")
    
    if overrideColormap:
        cmap=plt.cm.get_cmap(overrideColormap)
    
    if years==30:
        df = annTrends30yr5a
        if variable == "temperature":
            df = annTempTrends30yr5a
    elif years==50:
        df = annTrends50yr5a
        if variable == "temperature":
            df = annTempTrends50yr5a
    
    if subplotting:
        if overrideAspect:
            aspect = overrideAspect
        else:
            aspect = 5
        im = ax.imshow(array,cmap,vmin=-extent,vmax=extent,aspect=aspect)
        if extent < 0.001:
            cbar = fig.colorbar(im,ax=ax,extend="both",format='%.1e',shrink=0.8,label=units[variable])
        else:
            cbar = fig.colorbar(im,ax=ax,extend="both",shrink=0.8,label=units[variable])
        ticklabs = cbar.ax.get_yticklabels()
        cbar.ax.set_yticklabels(ticklabs, fontsize=8)
        yloc = np.arange(0,array.shape[0],5)
        ax.set_yticks(yloc)
        ax.set_yticklabels(altitudes[region].astype(int)[yloc])

        monthLoc = [0,31,59,90,120,151,181,212,243,274,304,335]
        months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        ax.set_xticks(monthLoc) 
        ax.set_xticklabels(months, fontsize=10,ha="left")
        
        if titleOverride:
            ax.set_title(titleOverride)
        else:
            ax.set_title(f"{regionLabels[region]}")

        # finding excluded catchments
        grey = []
        notgrey = []
        for i in range(array.shape[0]):
            if (array[i,:]==-99).all():
                grey.append(i)
            else:
                notgrey.append(i)
        # plotting excluded catchments in grey
        for i in grey:
            ax.axhline(i,color="whitesmoke",linewidth=4.7)
        
        #if variable != "temperature":
        try:
            field = np.array(FS[MA][years][var][region].fieldSignificant)
            x = np.arange(365).astype(float)
            y = np.full_like(x,-1.5)
            np.place(x,field==False,np.nan)
            np.place(y,field==False,np.nan)
            ax.plot(x,y,"-",color="gold",linewidth=4)
            ax.axhline(-0.8,linewidth=0.8,color="k")
            ax.set_ylim(array.shape[0],-2)
        except KeyError:
            print("Field significance not found. Plotting without FS... ")
            
        if variable == "streamflow":
            mag = np.array(df[df.region==region].runoff)
            mask = np.isnan(mag)
            if region=="vest" and years==50:
                mask = mask[:-1]
            y = np.array(notgrey).astype(float)
            x = np.full_like(y,367)
            np.place(y,mask,np.nan)
            for i in range(len(x)):
                if mag[i]<0:
                    ax.plot(x[i],y[i],"s",color="magenta",markersize=3)
                elif mag[i]>0:
                    ax.plot(x[i],y[i],"s",color="lime",markersize=3)
            ax.axvline(365.5,linewidth=0.8,color="k")
            ax.set_xlim(0,372)
        elif variable == "temperature":
            mag = np.array(df[df.region==region]["Ttrend_deg/yr"])
            mask = np.isnan(mag)
            if region=="vest" and years==50:
                mask = mask[:-1]
            y = np.array(notgrey).astype(float)
            x = np.full_like(y,367)
            np.place(y,mask,np.nan)
            for i in range(len(x)):
                if mag[i]<0:
                    ax.plot(x[i],y[i],"s",color="magenta",markersize=3)
                elif mag[i]>0:
                    ax.plot(x[i],y[i],"s",color="lime",markersize=3)
            ax.axvline(365.5,linewidth=0.8,color="k")
            ax.set_xlim(0,372)
        else:
            mag = np.array(df[df.region==region][variable])
            mask = np.isnan(mag)
            if region=="vest" and years==50:
                mask = mask[:-1]
            y = np.array(notgrey).astype(float)
            x = np.full_like(y,367)
            np.place(y,mask,np.nan)
            for i in range(len(x)):
                if mag[i]<0:
                    ax.plot(x[i],y[i],"s",color="magenta",markersize=3)
                elif mag[i]>0:
                    ax.plot(x[i],y[i],"s",color="lime",markersize=3)
            ax.axvline(365.5,linewidth=0.8,color="k")
            ax.set_xlim(0,372)
    
    else:
        if overrideAspect:
            aspect = overrideAspect
        else:
            aspect = 3
        plt.figure(figsize=(10,array.shape[0]/10))
        plt.imshow(array,cmap,vmin=-extent,vmax=extent,aspect=aspect)
        if extent<0.01:
            plt.colorbar(shrink=1/(array.shape[0])*30,
                         extend="both",
                         label=f"{units[variable]}",format='%.1e')
        else:
            plt.colorbar(shrink=1/(array.shape[0])*30,
                         extend="both",
                         label=f"{units[variable]}")
        monthLoc = [0,31,59,90,120,151,181,212,243,274,304,335]
        months = [" Jan"," Feb"," Mar"," Apr"," May"," Jun"," Jul"," Aug"," Sep"," Oct"," Nov"," Dec"]
        plt.xticks(monthLoc,months,ha="left")
        
        plt.ylabel("Altitude in $m.a.s.l.$")
        yloc = np.arange(0,array.shape[0],5)
        plt.yticks(yloc,altitudes[region].astype(int)[yloc])
        
        plt.title(f"{(v[variable]).upper()} {MA} {regionLabels[region]}")

        # finding excluded catchments
        grey = []
        notgrey = []
        for i in range(array.shape[0]):
            if (array[i,:]==-99).all():
                grey.append(i)
            else:
                notgrey.append(i)
        # plotting excluded catchments in grey
        for i in grey:
            plt.hlines(i,0,365,color="white",linewidth=4.7)
        
        #if variable != "temperature":
        try:
            field = np.array(FS[MA][years][variable][region].fieldSignificant)
            x = np.arange(365).astype(float)
            y = np.full_like(x,-1.5)
            np.place(x,field==False,np.nan)
            np.place(y,field==False,np.nan)
            plt.plot(x,y,"-",color="gold",linewidth=4)
            plt.hlines(-0.8,-3,370,linewidth=0.8)
            plt.ylim(array.shape[0],-2)
        except KeyError:
            print("Field significance not found. Plotting without FS... ")
            plt.ylim(array.shape[0],-1)
        
        if variable == "streamflow":
            mag = np.array(df[df.region==region].runoff)
            mask = np.isnan(mag)
            if region=="vest" and years==50:
                mask = mask[:-1]
            y = np.array(notgrey).astype(float)
            x = np.full_like(y,367)
            np.place(y,mask,np.nan)
            for i in range(len(x)):
                if mag[i]<0:
                    plt.plot(x[i],y[i],"s",color="magenta",markersize=3)
                elif mag[i]>0:
                    plt.plot(x[i],y[i],"s",color="lime",markersize=3)
            plt.vlines(365.5,-3,array.shape[0],linewidth=0.8)
            plt.xlim(0,369)
        elif variable == "temperature":
            mag = np.array(df[df.region==region]["Ttrend_deg/yr"])
            mask = np.isnan(mag)
            if region=="vest" and years==50:
                mask = mask[:-1]
            y = np.array(notgrey).astype(float)
            x = np.full_like(y,367)
            np.place(y,mask,np.nan)
            for i in range(len(x)):
                if mag[i]<0:
                    plt.plot(x[i],y[i],"s",color="magenta",markersize=3)
                elif mag[i]>0:
                    plt.plot(x[i],y[i],"s",color="lime",markersize=3)
            plt.vlines(365.5,-3,array.shape[0],linewidth=0.8)
            plt.xlim(0,369)
        else:
            mag = np.array(df[df.region==region][variable])
            mask = np.isnan(mag)
            if region=="vest" and years==50:
                mask = mask[:-1]
            y = np.array(notgrey).astype(float)
            x = np.full_like(y,367)
            np.place(y,mask,np.nan)
            for i in range(len(x)):
                if mag[i]<0:
                    plt.plot(x[i],y[i],"s",color="magenta",markersize=3)
                elif mag[i]>0:
                    plt.plot(x[i],y[i],"s",color="lime",markersize=3)
            plt.vlines(365.5,-3,array.shape[0],linewidth=0.8)
            plt.xlim(0,369)


        plt.savefig(f"Plots/Trends/{filename}_{variable}_{MA}_{region}.png",dpi=400,bbox_inches='tight')
        if not display:
            plt.close()

def plotFiles(files,filename,years,colormap=None):
    """
    Generates and saves plots for a list of files. 
    
    Parameters
    ----------
    files: list of filenames with path to file
    """

    for file in files:
        split = file[:-4].split("_")
        variable, region, MA = tuple(split[1:4])
        plotTrendMag(np.load(file),variable,region,MA,years,filename=filename,overrideColormap=colormap)
        

# ## Line plots

# ### Catchment plots

def catchmentLinePlot(data,region,MA,years,catchments = None, display = False):

    file = findFiles(f"streamflow_{region}",MA,years,head="Mag")[0]
    stream = np.load(file)
    file = findFiles(f"rainfall_{region}",MA,years,head="Mag")[0]
    rain = np.load(file)
    file = findFiles(f"snowmelt_{region}",MA,years,head="Mag")[0]
    snow = np.load(file)
    file = findFiles(f"temperature_{region}",MA,years,head="Ana")[0]
    temp = np.load(file)
    
    snumber = data[f"final30"]
    
    if years == 30:
        ANN = annTrends30yr5a
        ANNtemp = annTempTrends30yr5a
    elif years == 50:
        ANN = annTrends50yr5a
        ANNtemp = annTempTrends50yr5a

    monthLoc = [0,31,59,90,120,151,181,212,243,274,304,335]
    months = [" Jan"," Feb"," Mar"," Apr"," May"," Jun"," Jul"," Aug"," Sep"," Oct"," Nov"," Dec"]

    if catchments==None:
        catchments = np.arange(len(snumber))
    
    for c in catchments:
        if (temp[c,:]==-99).all():
            continue
        
        fig, ax1 = plt.subplots(figsize=(10,3))
        plt.xlim(0,365)
        plt.xticks(monthLoc, months, fontsize=12,ha="left") 
        
        arr = temp[c,:]
        index = np.where(arr>0)[0]
        plt.vlines(index,ymin=-10,ymax=10,color="whitesmoke",linewidth=3,label="Sign. pos. temperature trend")
        
        s = stream[c,:]
        ax1.plot(s,"k",label="Streamflow trend")
        lim = np.array([-np.nanmin(s),np.nanmax(s)]).max()*1.1
        ax1.set_ylim(-lim,lim)
        ax1.set_ylabel("Streamflow trend magnitude\n$m^3\ s^{-1}\ km^{-2}\ yr^{-1}$")

        ax2 = ax1.twinx()
        both = snow[c,:]+rain[c,:]
        ax2.plot(snow[c,:],":",color="grey",label="Snowmelt trend")
        ax2.plot(rain[c,:],":",color="skyblue",label="Rainfall trend")
        ax2.plot(both,"b--",label="Rainfall+snowmelt trend")
        lim = np.array([-np.nanmin(both),np.nanmax(both)]).max()*1.1
        ax2.set_ylim(-lim,lim)
        ax2.set_ylabel("Trend magnitude ($mm\ yr^{-1}$)\nRainfall and snowmelt")
        ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1e'))
        
        alt = int(altitudes[region][c])
        catchm = snumber[c]
        annRunoffTrend = float(ANN.runoff[ANN.snumber==catchm])
        annTempTrend = float(ANNtemp["Ttrend_deg/yr"][ANNtemp.snumber==catchm])
        
        if np.isfinite(annRunoffTrend):
            plt.title(f"Annual runoff trend: {annRunoffTrend:.2f} $mm\ yr^{-1}$",fontsize=10, loc="left")
        if np.isfinite(annTempTrend):
            plt.title(f"Annual temperature trend: {annTempTrend:.2f} {units['temperature']}",fontsize=10, loc="right")
        
        plt.savefig(f"Plots/Trends/Selected/{years}yearTrends_{MA}_{region}_{snumber[c]}.png",dpi=400,bbox_inches='tight')
    
        if not display:
            plt.close()



# ## Hydrographs
# * Calculate hydrograph for catchment based on 5 years before trend analysis
# * Extract "trend hydrograph" from trend magnitude array
# * Calculate "new" hydrograph

def plotHydrographs(MA, region, data, years, save=True, display=False):

    file = findFiles(MA=MA,variable=f"streamflow_{region}",years=years,head="Mag")[0]
    arr = np.load(file)

    for c in range(len(data[f"final{years}"])):
        catchment = data[f"final{years}"][c]

        area = data["metadata"][data["metadata"].snumber==catchment].areal.iloc[0]
        """if MA == "5day":
            ts = (data["data"][catchment]["runoff"]["1973":"1982"].runoff/area).rolling(5).mean()
        if MA == "10day":
            ts = (data["data"][catchment]["runoff"]["1973":"1982"].runoff/area).rolling(10).mean()
        if MA == "30day":
            ts = (data["data"][catchment]["runoff"]["1973":"1982"].runoff/area).rolling(30).mean()"""
        ts = (data["data"][catchment]["runoff"]["1973":"1982"].runoff/area)
        tsg = ts.groupby([ts.index.month,ts.index.day])
        means = []
        for name, group in tsg:
            means.append((group.mean()))
        HG = np.array(means[:-1])

        trendHG = arr[c,:]*years

        try:
            newHG = HG + trendHG
        except ValueError:
            newHG = HG + trendHG[:-1]


        monthLoc = [0,31,59,90,120,151,181,212,243,274,304,335]
        months = [" Jan"," Feb"," Mar"," Apr"," May"," Jun"," Jul"," Aug"," Sep"," Oct"," Nov"," Dec"]

        plt.figure(figsize=(10,4))
        plt.hlines(0,0,365,color="whitesmoke")
        plt.plot(HG,"k",label="Original hydrograph")
        plt.plot(trendHG,"r",label="Daily trend graph")
        plt.plot(newHG,"b--",label="New hydrograph")
        plt.legend()
        plt.xlim(0,365)
        plt.xticks(monthLoc, months, fontsize=12,ha="left") 
        plt.ylabel("Specific streamflow ($m^3\ s^{-1}\ km^{-2} $)")
        plt.title(f"{altitudes[region].astype(int)[c]} m.a.s.l.",loc="right")
        plt.title(f"{catchment} {regionLabels[region]}",loc="center")
        plt.title(f"{area:.0f} $km^2$",loc="left")

        if save:
            plt.savefig(f"Plots/Hydrographs/hydrograph_{years}year_{MA}_{region}_{catchment}.png",dpi=400,bbox_inches='tight')
        if not display:
            plt.close()


# ## Trend timing


def consecutive(data, stepsize=1):
    """
    This function has been taken directly from:
    https://stackoverflow.com/questions/7352684/how-to-find-the-groups-of-consecutive-elements-from-an-array-in-numpy
    by
    unutbu
    https://stackoverflow.com/users/190597/unutbu
    """
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)

def trendTiming(arr):
    """
    Calculates the trend timing according to Kormann et al. (2015)
    https://doi.org/10.2166/wcc.2014.099
    """
    index = np.where(np.isfinite(arr))[0]
    clusters = consecutive(index)
    
    timing = []
    for cluster in clusters:
        magnitudes = arr[cluster]
        moment = (magnitudes*cluster).sum()/magnitudes.sum()
        timing.append(moment)
    
    return timing

def trendTimingMinMax(arr):
    """
    Finds the central moment of trend cluster with minimum and maximum magnitude.
    
    Returns
    -------
    (timingMax,timingMin)
        tuple of trend timing for max and min
    """
    t = np.round(trendTiming(arr)).astype(int)
    mag = arr[t]
    imax = mag.argmax()
    imin = mag.argmin()
    return t[imax],t[imin]


def plotTrendClusters(Sarr,NSarr,subplotting=False,ax=None,colormap="cividis",size=None,var=None):
    N = Sarr.shape[0]
    cmap = plt.cm.get_cmap(colormap)
    norm = matplotlib.colors.Normalize(vmin=0,vmax=N)
    sm = plt.cm.ScalarMappable(cmap=cmap,norm=norm)
    sm.set_array([])
    rcParams['axes.prop_cycle'] = cycler(color=cmap(np.linspace(0, 1, N)))
    
    if subplotting:
        #for c in range(N):
            #ax.plot(NSarr[c,:],"--",linewidth=1,alpha=0.5)
        for c in range(N):
            ax.plot(Sarr[c,:],linewidth=2,alpha=0.5)
    else:
        plt.figure(figsize=size)
        #plt.title("Trend magnitude",loc="right")
        plt.ylabel(f"Trend magnitude ({units[var]})")
        plt.xlabel("DOY")
        plt.axhline(0,color="grey")
        plt.xlim(0,365)
        for c in range(N):
            plt.plot(NSarr[c,:],"--",linewidth=1,alpha=0.5)
        for c in range(N):
            plt.plot(Sarr[c,:],linewidth=2,alpha=0.5)
        cbar = plt.colorbar(sm,label="Station by altitude")
        cbar.ax.invert_yaxis()



def plotPosNegTrendTiming(arr,altitudes,subplotting=False,ax=None):
    cmap = plt.cm.get_cmap("cividis",2)
    pos = cmap(0)
    neg = "goldenrod"
    if subplotting:
        for c in range(arr.shape[0]):
            subArr = arr[c,:]
            x = np.round(trendTiming(subArr)).astype(int)
            x = np.delete(x,np.where(x<0))
            y = altitudes[c]
            for i in range(len(x)):
                t = x[i]
                if subArr[t]>0:
                    ax.plot(x[i],y,"^",color=pos)
                else:
                    ax.plot(x[i],y,"v",color=neg)
    else:
        plt.figure()
        plt.xlim(0,365)
        plt.title("Trend timing",loc="right")
        plt.xlabel("DOY")
        plt.ylabel("Altitude in $m.a.s.l.$")
        for c in range(arr.shape[0]):
            subArr = arr[c,:]
            x = np.round(trendTiming(subArr)).astype(int)
            x = np.delete(x,np.where(x<0))
            y = altitudes[c]
            for i in range(len(x)):
                t = x[i]
                if subArr[t]>0:
                    plt.plot(x[i],y,"^",color=pos)
                else:
                    plt.plot(x[i],y,"v",color=neg)



def subplotTrendClusterTiming(Sstream,Ssnow,Srain,NSstream,NSsnow,NSrain,region,MA,years):
    fig,ax = plt.subplots(nrows=2,ncols=3,sharex=True,figsize=(15,7))
    # streamflow
    plotTrendClusters(Sstream,NSstream,subplotting=True,ax=ax[0][0])
    plotPosNegTrendTiming(Sstream,altitudes[region],subplotting=True,ax=ax[1][0])
    # snowmelt
    plotTrendClusters(Ssnow,NSsnow,subplotting=True,ax=ax[0][1])
    plotPosNegTrendTiming(Ssnow,altitudes[region],subplotting=True,ax=ax[1][1])
    # rainfall
    plotTrendClusters(Srain,NSrain,subplotting=True,ax=ax[0][2])
    plotPosNegTrendTiming(Srain,altitudes[region],subplotting=True,ax=ax[1][2])
    # labels
    ax[0][0].set_ylabel("Trend Magnitude")
    ax[1][0].set_ylabel("Altitude in $m.a.s.l.$")
    ax[0][0].set_title("Streamflow")
    ax[0][1].set_title("Snowmelt")
    ax[0][2].set_title("Rainfall")
    ax[1][1].set_xlabel("DOY")
    
    N = Sstream.shape[0]
    cmap = plt.cm.cividis
    norm = matplotlib.colors.Normalize(vmin=0,vmax=N)
    sm = plt.cm.ScalarMappable(cmap=cmap,norm=norm)
    sm.set_array([])
    
    cmap = plt.cm.get_cmap("cividis",2)
    pos = cmap(0)
    neg = "goldenrod"
    legend_elements = [Line2D([0], [0], marker='^', color='w', label="Positive",markerfacecolor=pos, markersize=10),
                       Line2D([0], [0], marker='v', color='w', label="Negative",markerfacecolor=neg, markersize=10),
                       Line2D([0], [0], marker='^', color='w', label="cluster",markerfacecolor="w", markersize=10)]
    ax[1][2].legend(handles=legend_elements, loc="right",bbox_to_anchor=(1.4, 0.5),title="Trend timing of")
    
    cbar_ax = fig.add_axes([0.92, 0.55, 0.01, 0.3])
    cbar = fig.colorbar(sm,cax=cbar_ax,label=f"Station by altitude")
    cbar.ax.invert_yaxis()
    # save plot
    plt.savefig(f"Plots/TrendTiming_{years}year_{MA}_{region}.png",dpi=400,bbox_inches='tight')



def findPlotSave(region,MA,years):
    file = findFiles(head="Ana",variable=f"streamflow_{region}",MA=MA,years=years)[0]
    arr = np.load(file)
    np.place(arr,arr==-99,np.nan)
    Sstream = arr
    
    file = findFiles(head="Mag",variable=f"streamflow_{region}",MA=MA,years=years)[0]
    arr = np.load(file)
    np.place(arr,arr==-99,np.nan)
    NSstream = arr

    file = findFiles(head="Ana",variable=f"snowmelt_{region}",MA=MA,years=years)[0]
    arr = np.load(file)
    np.place(arr,arr==-99,np.nan)
    Ssnow = arr
    
    file = findFiles(head="Mag",variable=f"snowmelt_{region}",MA=MA,years=years)[0]
    arr = np.load(file)
    np.place(arr,arr==-99,np.nan)
    NSsnow = arr

    file = findFiles(head="Ana",variable=f"rainfall_{region}",MA=MA,years=years)[0]
    arr = np.load(file)
    np.place(arr,arr==-99,np.nan)
    Srain = arr
    
    file = findFiles(head="Mag",variable=f"rainfall_{region}",MA=MA,years=years)[0]
    arr = np.load(file)
    np.place(arr,arr==-99,np.nan)
    NSrain = arr

    subplotTrendClusterTiming(Sstream,Ssnow,Srain,NSstream,NSsnow,NSrain,region,MA=MA,years=years)


def findPlotSingle(region,var,MA,years,size=None,savefile=False,display=False):
    Sfile = findFiles(head="Ana",variable=f"{var}_{region}",MA=MA,years=years)[0]
    NSfile = findFiles(head="Mag",variable=f"{var}_{region}",MA=MA,years=years)[0]

    Sarr = np.load(Sfile)
    np.place(Sarr,Sarr==-99,np.nan)
    NSarr = np.load(NSfile)
    np.place(NSarr,NSarr==-99,np.nan)

    plotTrendClusters(Sarr,NSarr,size=(10,5),var=var)
    
    if savefile:
        plt.savefig(f"Plots/Trends/Line/TrendLine_{region}_{var}_{MA}_{years}year.png",dpi=400,bbox_inches='tight')
    
    if not display:
        plt.close()


def plotAltMag(region,var,MA,years,withMinMax=False,subplotting=False,ax=None,legend=False,head="Ana"):
    file = findFiles(head=head,variable=f"{var}_{region}",MA=MA,years=years)[0]
    array = np.load(file)
    
    height = array.shape[0]/10
    
    mask = np.unique(np.where(array!=-99)[0])
    array = array[mask,:]
    
    alt = altitudes[region][mask]
    
    m = np.nanmean(array,axis=1)
    s = np.nanstd(array,axis=1)
    
    mini = []
    maxi = []
    for c in range(array.shape[0]):
        mini.append(np.nanmin(array[c,:]))
        maxi.append(np.nanmax(array[c,:]))
    
    if subplotting:
        ax.axvline(0,color="k",linewidth=1)
        
        if withMinMax:
            ax.fill_betweenx(alt,mini,maxi,color="lightgrey",alpha=0.5,label="Minimum-maximum values")
        
        ax.fill_betweenx(alt,m-s,m+s,color="grey",alpha=0.5,label="Standard deviation")

        ax.plot(m,alt,"b-",label="Mean")
        ax.set_title(f"n = {array.shape[0]}")
        ax.set_ylabel(regionLabels[region],fontsize=14)
        #ax.set_ylim(0,alt.max())
        
        if var == "streamflow":
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.0e'))
        
        if legend:
            ax.legend(loc="best")
    else:
        plt.figure(figsize=(5,height))
        plt.vlines(0,ymin=0,ymax=alt.max(),color="k",linewidth=1)
        plt.ylim(0,alt.max())
        if withMinMax:
            plt.fill_betweenx(alt,mini,maxi,color="lightgrey",alpha=0.5,label="Minimum-maximum values")

        plt.fill_betweenx(alt,m-s,m+s,color="grey",alpha=0.5,label="Standard deviation")

        plt.plot(m,alt,"b-",label="Mean")
        plt.title(regionLabels[region])
        plt.xlabel(f"Trend magnitude ({units[var]})")
        plt.ylabel("Altitude in $m.a.s.l.$")
        if legend:
            plt.legend(loc="best")

def plotAltMagPercentile(region,var,q,MA="10day",years=30,ax=None,head="Ana"):
    file = findFiles(head=head,variable=f"{var}_{region}",MA=MA,years=years)[0]
    array = np.load(file)
    
    mask = np.unique(np.where(array!=-99)[0])
    array = array[mask,:]
    
    alt = altitudes[region][mask]

    high = np.nanpercentile(array,q,axis=1)
    reg = stats.linregress(alt,high)
    x = np.linspace(alt.min(),alt.max())
    
    
    if ax:
        ax.plot(high,alt,"kx")
        
        if reg.pvalue<0.05:
            ax.plot(x*reg.slope+reg.intercept,x,color="k")
            mod = alt*reg.slope+reg.intercept
            R = ((mod-high)**2).mean()
            if var == "streamflow":
                ax.set_title(f"$RMSE$ = {R:.1e}",loc="right")
            else:
                ax.set_title(f"$RMSE$ = {R:.4f}",loc="right")
        else:
            ax.plot(x*reg.slope+reg.intercept,x,"--",color="grey")
            mod = alt*reg.slope+reg.intercept
            R = ((mod-high)**2).mean()
            if var == "streamflow":
                ax.set_title(f"$RMSE$ = {R:.2e}",loc="right",color="grey")
            else:
                ax.set_title(f"$RMSE$ = {R:.4f}",loc="right",color="grey")
        
        if (region == "vest" or region=="ost") and (var=="snowmelt" or var=="rainfall"):
            p = np.polyfit(alt,high,2)
            mod = p[0]*alt**2+p[1]*alt+p[2]
            RMSE = ((mod-high)**2).mean()
            y = np.linspace(alt.min(),alt.max())
            ax.plot(p[0]*y**2+p[1]*y+p[2],y,color="r")
            if var == "streamflow":
                ax.set_title(f"$RMSE$ = {RMSE:.2e}",loc="left",color="r")
            else:
                ax.set_title(f"$RMSE$ = {RMSE:.4f}",loc="left",color="r")
        
        if var == "streamflow":
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.0e'))
        
        if region=="finn":
            ax.set_title(f"{q}th percentile", loc="left")
        
    else:
        plt.figure()
        plt.plot(high,alt,"kx")
        
        if reg.pvalue<0.05:
            plt.plot(x*reg.slope+reg.intercept,x,color="k")
            mod = alt*reg.slope+reg.intercept
            R = ((mod-high)**2).mean()
            if var == "streamflow":
                plt.title(f"$RMSE$ = {R:.1e}",loc="right")
            else:
                plt.title(f"$RMSE$ = {R:.4f}",loc="right")
        else:
            plt.plot(x*reg.slope+reg.intercept,x,"--",color="grey")
            mod = alt*reg.slope+reg.intercept
            R = ((mod-high)**2).mean()
            if var == "streamflow":
                plt.title(f"$RMSE$ = {R:.1e}",loc="right",color="grey")
            else:
                plt.title(f"$RMSE$ = {R:.4f}",loc="right",color="grey")
        
        if (region == "vest" or region=="ost") and (var=="snowmelt" or var=="rainfall"):
            p = np.polyfit(alt,high,2)
            mod = p[0]*alt**2+p[1]*alt+p[2]
            RMSE = ((mod-high)**2).mean()
            y = np.linspace(alt.min(),alt.max())
            plt.plot(p[0]*y**2+p[1]*y+p[2],y,color="r")
            if var == "streamflow":
                plt.title(f"$RMSE$ = {RMSE:.1e}",loc="left",color="r")
            else:
                plt.title(f"$RMSE$ = {RMSE:.4f}",loc="left",color="r")
        
        plt.xlabel(units[var])
        plt.title(regionLabels[region])



def AnnualvsDailyTrendSum(var,ax=None,years=30,head="Mag"):
    regions = ("finn","nord","trond","vest","ost","sor")
    if years==30:
        annTrends = annTrends30yr5a
        annTempTrends = annTempTrends30yr5a
    elif years==50:
        annTrends = annTrends50yr5a
        annTempTrends = annTempTrends50yr5a
    plt.figure(figsize=(4,4))
    for region in regions:
        file = findFiles(head=head,variable=f"{var}_{region}",MA=MA,years=years)[0]
        array = np.load(file)
        mask = np.unique(np.where(array!=-99)[0])
        array = array[mask,:]
        alt = altitudes[region][mask]
        dailySum = np.nansum(array,axis=1)
        
        if var == "temperature":
            ANN = np.array(annTempTrends["Ttrend_deg/yr"][annTempTrends.region==region])
            dailySum = np.nanmean(array,axis=1)
        elif var == "streamflow":
            ANN = np.array(annTrends["runoff"][annTrends.region==region])
        else:
            ANN = np.array(annTrends[var][annTrends.region==region])
        
        mini = np.nanmin([np.nanmin(ANN),np.nanmin(dailySum)])
        maxi = np.nanmax([np.nanmax(ANN),np.nanmax(dailySum)])
        
        
        if var!="streamflow":
            plt.plot([-30,30],[-30,30],color="grey")
            plt.xlim(mini,maxi)
            plt.ylim(mini,maxi)
        plt.plot(dailySum,ANN,"k.")
         
        if ax:
            ax.plot(dailySum,ANN,"k.")
            ax.set_ylabel("Daily trend sum")
            ax.set_xlabel("Annual trend")

def dailyTrendSumvsAltitude(region,MA="10day",years=30,head="Mag",ax=None):
    file = findFiles(head=head,variable=f"{var}_{region}",MA=MA,years=years)[0]
    array = np.load(file)

    mask = np.unique(np.where(array!=-99)[0])
    array = array[mask,:]

    alt = altitudes[region][mask]
    dailyTrendSum = np.nansum(array,axis=1)
    reg = stats.linregress(alt,dailyTrendSum)
    x = np.linspace(alt.min(),alt.max())
    
    if ax:
        ax.plot(dailyTrendSum,alt,".",color="green")
        ax.plot(x*reg.slope+reg.intercept,x,"-",color="green")
    else:
        plt.plot(dailyTrendSum,alt,".",color="green")
        plt.plot(x*reg.slope+reg.intercept,x,"-",color="green")


def trendCorrelation(head = "Mag",MA = "10day",years = 30, maskSmall=None, correlateWith = "both"):
    qtrend = []
    srtrend = []
    regions = ("finn","nord","trond","vest","ost","sor")
    for region in regions:
        file = findFiles(head=head,variable=f"streamflow_{region}",MA=MA,years=years)[0]
        arr = np.load(file)
        mask = np.unique(np.where(arr!=-99)[0])
        arr = arr[mask,:]
        stream = arr

        file = findFiles(head=head,variable=f"rainfall_{region}",MA=MA,years=years)[0]
        arr = np.load(file)
        mask = np.unique(np.where(arr!=-99)[0])
        arr = arr[mask,:]
        rain = arr

        file = findFiles(head=head,variable=f"snowmelt_{region}",MA=MA,years=years)[0]
        arr = np.load(file)
        mask = np.unique(np.where(arr!=-99)[0])
        arr = arr[mask,:]
        snow = arr
    
        if correlateWith == "both":
            for c in range(stream.shape[0]):
                q = stream[c,:]
                qtrend += list(q)
                d = snow[c,:] + rain[c,:]
                srtrend += list(d)
        elif correlateWith == "snow":
            for c in range(stream.shape[0]):
                q = stream[c,:]
                qtrend += list(q)
                d = snow[c,:]
                srtrend += list(d)
        elif correlateWith == "rain":
            for c in range(stream.shape[0]):
                q = stream[c,:]
                qtrend += list(q)
                d = rain[c,:]
                srtrend += list(d)
            
    plt.figure(figsize=(5,5))
    plt.plot(qtrend,srtrend,"x",color="lightgrey")
    qtrend
    reg1 = stats.linregress(qtrend,srtrend)
    x = np.linspace(np.min(qtrend),np.max(qtrend))
    
    plt.title(f"$r$ = {reg1.rvalue:.2f}",loc="right")
    #plt.title(regionLabels[region])
    plt.xlabel(f'{units["streamflow"]}\nStreamflow trend')
    
    if correlateWith == "both":
        plt.ylabel(f"Snowmelt + rainfall trend\n{units['rainfall']}")
    elif correlateWith == "snow":
        plt.ylabel(f"Snowmelt trend\n{units['rainfall']}")
    elif correlateWith == "rain":
        plt.ylabel(f"Rainfall trend\n{units['rainfall']}")
    
    qtrend = np.array(qtrend)
    srtrend = np.array(srtrend)
    
    if maskSmall:
        mask = ~((qtrend<maskSmall) & (qtrend>-maskSmall))
        qtrend = qtrend[mask]
        srtrend = srtrend[mask]
    
    plt.plot(qtrend,srtrend,"+",color="grey")
    reg = stats.linregress(qtrend,srtrend)
    x = np.linspace(np.min(qtrend),np.max(qtrend))
    plt.plot(x,x*reg1.slope+reg1.intercept,"k",linewidth=3)
    plt.plot(x,x*reg.slope+reg.intercept,"b",linewidth=3)
    plt.title(f"$r$ = {reg.rvalue:.2f}",loc="left",color="b")


def sumDays(arr,N=5):
    """
    N day averages of array
    """
    averaged = []
    for i in range(int(arr.shape[1]/N)):
         averaged.append(np.nansum(arr[:,i*N:i*N+N],axis=1))
    return np.array(averaged).T

