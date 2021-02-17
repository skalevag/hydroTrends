import matplotlib.pyplot as plt
import numpy as np

def plotTrendArray(ta,factor = 10,cmap = "RdBu",plotSignificance=True,colorbarLimit=0.8):
    """
    Plot trend array.

    Parameters
    ----------
    ta: trendArray object
    factor: int
        default: 10
    cmap: str
        default: "RuBu
        name of colormap
        should be a diverging colomap to show positive and negative trends
    plotSignificance: boolean
        default: True
    colorbarLimit: float
        default: 0.8
    """
    # magnitudes
    arr = ta.magnitudes*factor
    extent = round(np.quantile(arr,colorbarLimit,axis=1).max())

    # plot
    plt.figure(figsize=(15,arr.shape[0]/2))
    plt.pcolormesh(arr,vmin=-extent,vmax=extent,cmap=cmap)

    # colorbar
    unit = f"{ta.trendUnit.split('/')[0]}/ {factor} yr"
    plt.colorbar(shrink=5/arr.shape[0],extend="both",label=unit)

    # significance
    try:
        sig = ta.significance
        if plotSignificance:
            mask = np.ma.masked_array(np.full_like(sig,-9999),sig == 1)
            plt.pcolor(mask, hatch='/////', alpha=0.,snap=True,label="Non-significant")
    except AttributeError:
        pass
    

    # field significance
    # TODO
    #plt.hlines(-0.1,0,365,colors="k",linewidth=0.5)
    #field = data[region][var]["FieldSig"]
    #x = np.arange(365).astype(float)
    #y = np.full_like(x,-0.5)
    #np.place(x,field==False,np.nan)
    #np.place(y,field==False,np.nan)
    #plt.plot(x,y,"-",color="orange",linewidth=4)

    # labels etc
    plt.title(f"{ta.tsStack.variable.capitalize()} trend")
    plt.xlabel("DOY")
    plt.ylabel(f"{ta.sortedBy.capitalize()} [{ta.tsStack.sortAttributeUnit}]")
    ticks = np.arange(0,arr.shape[0],1)
    plt.yticks(ticks+0.5,np.array(ta.tsStack.sortAttributeValues))
    plt.ylim(arr.shape[0],0)

