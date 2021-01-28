## Introduction

This is an automatic asteroseismic pipeline to extract global seismic parameters from the light curve / power spectrum of a solar-like oscillating star. 

There are several parts in the pipeline:

- Light curve preparation: remove outliers & high-pass filter. 
- Find the power excess by collapsed autocorrelation function. 
- Derive global oscillation parameters and their uncertainties. A Bayesian fit to the power spectrum for $\nu_{max}$ and autocorrelation function for $\Delta \nu$. 



## Package Requirements

`numpy`, `matplotlib`, `scipy`, `astropy`, `emcee`, `corner` 



## The File Tree

### data

- This directory contains raw light curves to be processed. 

### output

- This directory contains all files outputted by the code. 

### targetlist.csv

- IDs and basic parameters of stars to be processed. 

### main.py

- Main part of the code.

### driver.py

- Where to run the code.  



## Usage

1. Git clone to a local directory. 
2. Simply run the script. 

```bash
python driver.py
```

 

More details will come soon. 

