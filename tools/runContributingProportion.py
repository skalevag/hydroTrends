import HTfunctions as ht #ht as in hydroTrends
import numpy as np
import datetime as dt
import pandas as pd


# functions for calculating contributing proportion

def contributionToRunoff(s,days):
    """
    Calculating the contribution of a variable (rainfall or snowmelt) to runoff in mm
    
    Parameters
    ----------
    s: pandas.Series
        data series containing daily data in mm/d
    days: int
        concentration/recession time in days
        
    Returns
    -------
    dataframe with contribution of snowmelt/rainfall to in mm per tc days
    """
    first = days-1
    t = dt.timedelta(days=days-1)

    dates = []
    contribution = []

    for d in range(first,s.shape[0]):
        start = s.index.date[d]-t
        end = s.index.date[d]

        sliced = s[start:end]
        day = sliced.index.date[-1]
        cont = sliced.sum()
        dates.append(day)
        contribution.append(cont)
    df = pd.DataFrame({"date":dates, "cont":contribution})
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")

def contributingProportion(sCont,rCont):
    """
    Calcualtes the contribution proportion of snowmelt/rainfall to runoff.
    
    proportion = contribution / total contribution
    
    Parameters
    ----------
    sCont: pandas.DataFrame
        snowmelt contribution
    rCont: pandas.DataFrame
        rainfall contribution
    
    Returns
    -------
    
    """
    totCont = sCont+rCont
    sContFraq = sCont / totCont
    sContFraq[sContFraq.isnull()]=0
    rContFraq = rCont / totCont
    rContFraq[rContFraq.isnull()]=0
    
    return sContFraq, rContFraq


def contributionSingleCatchment(c,data):
    """
    Calculates the contribution of snowmelt and rainfall to runoff for a specified catchment. 
    
    Parameters
    ----------
    data: dict
        dictionary containing the (final) data for a region,
        with sub-dictionaries for each catchment
    Returns
    -------
    
    """
    s = data["data"][c]["snow"].qsw #snowmelt
    r = data["data"][c]["precip"] #rainfall
    
    if c in list(durationDaysMod.snumber):
        tc = int(durationDaysMod[durationDaysMod.snumber==c]["conc.time"])
        tr = int(durationDaysMod[durationDaysMod.snumber==c]["rec.time"])
    elif c in list(durationDaysEmp.snumber):
        tc = 2
        tr = int(durationDaysEmp[durationDaysEmp.snumber==c]["recess_days"])
    elif c in list(durationDaysEmp2.snumber):
        tc = 2
        tr = int(durationDaysEmp2[durationDaysEmp2.snumber==c]["recess_days"])
    else:
        raise Exception(f"Parameter for catchment {c} could not be found.")
    
    sCont = contributionToRunoff(s,tc+tr)
    rCont = contributionToRunoff(r,tc+tr)
    
    return sCont,rCont


def saveToFile(contProp,snumber,variable,folder="Data/contributingProportion"):
    startYear = str(contProp.index.year[0])
    endYear = str(contProp.index.year[-1])
    contProp.to_csv(f"{folder}/{snumber}_contProp_{variable}_{startYear}_{endYear}.csv")


def contributionRegion(data, folder="Data/contributingProportion", overwrite = False):
    """
    Calculates the contribution proportion of snowmelt and rainfall to runoff
    for all catchments in a region. 
    
    Parameters
    ----------
    data: dict
        dictionary containing the (final) data for a region
    overwrite: bool
        {True, False}
        if .csv files already exist for a catchment the file is overwritten
    
    Returns
    -------
    
    """
    
    catchments = data["order"]
    N = len(catchments)
    print("Analysing",N,"catchments.")

    for i,c in enumerate(catchments,1):
        print(f"{i}/{N}")
        a = ht.findFiles(f"*{c}_*.csv",folder)
        
        if len(a) == 0:
            print(f"No .csv file exists for {c}. Calculating now...")
            try:
                cont = contributionSingleCatchment(c,data)
            except KeyError:
                print("Opsi, something is wrong!")
                print("KeyError, likely caused by problem with dates in original file.")
                print(f"Skipping {c}")
                continue
            sCont = cont[0]
            rCont = cont[1]
            
            contProp = contributingProportion(sCont,rCont)
            
            sContProp = contProp[0]
            rContProp = contProp[1]
            
            saveToFile(sContProp,c,"sm")
            saveToFile(rContProp,c,"rf")

            #print("Done.")
        
        elif overwrite:
            print(f"File already exists for {c}, but overwrite is activated. Calculating again...")
            try:
                cont = contributionSingleCatchment(c,data)
            except:
                print("Opsi, something is wrong...")
                continue
            sCont = cont[0]
            rCont = cont[1]
            
            contProp = contributingProportion(sCont,rCont)
            
            sContProp = contProp[0]
            rContProp = contProp[1]
            
            saveToFile(sContProp,c,"sm")
            saveToFile(rContProp,c,"rf")
        
        else:
            print(f"File already exists for {c}. Skipping catchment...")




folder="Data/contributingProportion"
print("Starting...")
# importing unaltered streamflow, snowmelt and rainfall data from dictionaries
ost = ht.openDict("Data/ostlandet_final.pkl")
vest = ht.openDict("Data/vestlandet_final.pkl")
nord = ht.openDict("Data/nordland_final.pkl")
trond = ht.openDict("Data/trondelag_final.pkl")
sor = ht.openDict("Data/sorlandet_final.pkl")
finn = ht.openDict("Data/finnmark_final.pkl")

# importing the concentration/recession time parameters
durationDaysMod = pd.read_csv("Data/critical_flood_durations_extended.txt",encoding="ISO-8859–1",sep=" ")
durationDaysEmp = pd.read_csv("Data/recess_lamda_0_99.txt",sep="\t")
durationDaysEmp2 = pd.read_csv("Data/recess_lamda_0_995.txt",sep="\t")

# removing empty rows
mask = durationDaysEmp.snumber.isna()
durationDaysEmp = durationDaysEmp[~mask]
mask = durationDaysEmp2.snumber.isna()
durationDaysEmp2 = durationDaysEmp2[~mask]

# making "snumber" out of "stnr"
regine = []
main = []
snumber = []
for i in range(durationDaysMod.shape[0]):
    string = str(durationDaysMod.stnr.iloc[i].round(3))
    split = string.split(".")
    r = split[0]
    m = "{:>03}".format(split[1])
    sn = r+"00"+m
    regine.append(int(r))
    main.append(int(m))
    snumber.append(int(sn))

durationDaysMod.insert(0,"main",main)
durationDaysMod.insert(0,"regine",regine)
durationDaysMod.insert(0,"snumber",snumber)

print("Prep finished.\n\nStarting calculations...")

# calculating the contributing proportions for both regions
print("Calculating Ostlandet...")
contributionRegion(ost,overwrite=False)
print("Ostlandet finished.\n")

print("Calculating Vestlandet...")
contributionRegion(vest,overwrite=False)
print("Vestlandet finished.\n")

print("Calculating Nordland...")
contributionRegion(nord,overwrite=False)
print("Nordland finished.\n")

print("Calculating Trondelag...")
contributionRegion(trond,overwrite=False)
print("Trondelag finished.\n")

print("Calculating Sorlandet...")
contributionRegion(sor,overwrite=False)
print("Sorlandet finished.\n")

print("Calculating Finnmark...")
contributionRegion(finn,overwrite=False)
print("Finnmark finished.\n")

print("Calculations completed and saved.")
