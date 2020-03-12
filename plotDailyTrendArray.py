"""
Script for plotting a trend array.
"""

from HTfunctions import *
import matplotlib.pyplot as plt
from matplotlib import cycler,rcParams,colors
import os
import pandas as pd
import dailyTrendPlottingFunctions as trendPlot
import sys


period = inFile = sys.argv[1]
var = inFile = sys.argv[2]
region = inFile = sys.argv[3]

try:
    file = findFiles(f"*Mag*{var}*{region}*{period}*","Results/Daily")[0]
    print(f"Creating plots for:\n{file}\n")

    # checking if plot directory exists for period
    # if none exists, a directory is created
    out_dir = f"Plots/{period}"
    if not os.path.isdir(out_dir):
        print(f"Directory {out_dir} didn't exist. Creating now...")
        os.mkdir(out_dir)
        print("Done.")
    else:
        print(f"Directory {out_dir} already exists.")
    print(f"All plots are saved in {out_dir}")

    # getting the name of file
    a = file.split("/")[-1].split(".")[0].split("_")[1:]
    period = "-".join(a[-2:])
    print("Period:",period)
    var = a[0]
    print("Variable:",var)
    region = a[1]
    MA = a[2]
    name = "_".join(a)
    print(name)


    # loading the trend array
    arr = np.load(file)
    print("\nTrend array shape:",arr.shape)


    finalCatchments = openDict("Data/finalSelectionList")
    catchments = finalCatchments[region][period]
    print("\nCatchments analysed:")
    for c in catchments:
        print(c)

    # metadata
    meta = pd.read_csv("Data/updated_stationselection.csv",index_col=0)

    # extracting altitudes
    altitudes = {}
    for c in catchments:
        altitudes[c] = int(meta[meta.snumber==c].altitude)



    # plotting the trend array
    plt.figure(figsize=(15,2))
    extent = np.array([np.abs(arr.min()),np.abs(arr.max())]).max()
    plt.imshow(arr,aspect=3,cmap="seismic_r",vmax=extent,vmin=-extent)
    plt.ylabel("Altitude")
    yloc = np.arange(0,arr.shape[0],2)
    plt.yticks(yloc,np.array(list(altitudes.values()))[yloc])
    plt.ylim(arr.shape[0]+0.5,-0.5)
    plt.colorbar(shrink=0.8,label="Streamflow trend\n$m^{3} s^{-1} \ yr^{-1}$")
    plt.title(f"{var.upper()} {period}")
    plt.xlabel("Day of year")
    plt.savefig(f"{out_dir}/dailyTrend_{name}.png",dpi=400,bbox_inches='tight')


    N = arr.shape[0]
    cmap = plt.cm.get_cmap("copper_r")
    norm = colors.Normalize(vmin=0,vmax=N)
    sm = plt.cm.ScalarMappable(cmap=cmap,norm=norm)
    sm.set_array([])
    rcParams['axes.prop_cycle'] = cycler(color=cmap(np.linspace(0, 1, N)))

    plt.figure(figsize=(20,7))
    for c in range(arr.shape[0]):
        catch = catchments[c]
        plt.plot(arr[c,:],label=f"{altitudes[catch]} m.a.s.l. ({catch})")
    plt.legend()
    plt.savefig(f"{out_dir}/lineTrend_{name}.png",dpi=400,bbox_inches='tight')

except IndexError:
    print("Can't find any file matching parameters in directory.")