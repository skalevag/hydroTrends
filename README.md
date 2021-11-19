# hydroTrends: A tool for high-resolution trend analysis
This module was developed for the scientific paper [Skålevåg & Vormoor, 2021](https://doi.org/10.1002/hyp.14329).

It contains two main tools:
* Trend analysis tool
* Plotting tool (in development)

For a brief demonstration of the tool see [here](demo.ipynb).

## Dependencies
For the Mann-Kendall test and Theil-Sen Slope Estimator the [USGS *trend* module](https://github.com/USGS-python/trend), in addition to a number of other [required modules](requirements.txt)

## Trend analysis
Daily trend analysis procedure developed by [Kormann et al., 2014](https://doi.org/10.2166/wcc.2014.099) based on [Déry et al., 2009](https://doi.org/10.1029/2008WR006975). See [Skålevåg & Vormoor, 2021](https://doi.org/10.1002/hyp.14329) for example application of this tool.

## Plotting (in development)
My own code for plotting the highly resolved trends, as used in [Skålevåg & Vormoor, 2021](https://doi.org/10.1002/hyp.14329).

