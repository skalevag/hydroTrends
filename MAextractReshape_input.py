"""
This script specifies the parameters used in MAextractReshape.py
"""
import HTfunctions as ht

final = ht.openDict("Data/finalSelectionList.pkl")

## parameters
var = "streamflow"
MA = 10
folder = "Data/runoff" # location of timeseries files 
region = "vest"
startYear = 1983
endYear = 2012
catchments = final[region]["1983-2012"]
destinationFolder = "Data/Reshaped"
function = ht.readRunoff # for loading timeseries data