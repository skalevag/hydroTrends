import numpy as np
import sys

## PARAMETERS AND SETUP
inFile = sys.argv[1]
filename = inFile.split("/")[-1]
name = filename.split(".")[0]
print("Analysing the file",filename)

out_dir = "Results/dailyMeans"
try: 
    print(sys.argv[2], "is set as output directory.")
    out_dir = sys.argv[2]
except IndexError:
    print(out_dir, "is set as output directory.")

array = np.load(inFile)

doyMean = np.nanmean(array,axis=1)

print("Averaged array has shape:",doyMean.shape)

np.save(f"{out_dir}/doyMean_{name}.npy", doyMean)

print(f"Averaged array can be found in {out_dir} directory.")