# **canopy_foss** 

A work in progress open source canopy and deforestation monitoring. Parallel
research to the [CanoPy](https://github.com/HuidaeCho/canopy) module created
for the Georgia Canopy Analysis 2009 project sponsored by the Georgia Forestry
Commission

## Overview

`canopy_foss` is a python module created to process large amounts of NAIP
imagery and create accurate canopy classifications in an open source
framework. Need for an open source classification system arose during the
creation of the Georgia canopy dataset as tools that were being used
, ArcMap and Textron's Feature Analyst, will be phased out within the next
few years. Additionally need for open source arose out of the lack of
insight to the algorithms that were being used by the software to
process our data and no true method to tweak it to suit our needs.

## Index functions

All index functions are stored in `canopy_foss.indicies.py`

The main focus will be on utilizing RGB index calculations as opposed to
the standard nir band vegetation calculations that are standard. This is due
to the wider availability of 3 band NAIP imagery as most 4 band (RGB + NIR
) NAIP imagery is not accessible without paying. 

**ARVI: ~ NIR based index** 

* SSD computation time: 1.45 seconds per calculation or approx. 1:35 hours 
  for full computation of all 3,913 ga naip tiles to NVME SSD. 

* HDD computation time: 6.16 hours hours to process all 3,913 ga naip tiles

**nVARI: ~ current focus** 

* HDD computation time: 5:45 hours to process all 3,913 ga naip tiles

* nVARI bands are normalized between -1 and 1 before being computed as the VARI
  formula [(green - red) / (green + red - blue)] ensures the values do not
  inherently fall between -1 and 1. Normalization ensures consistency for data
  across the entire NAIP data set. 

**GRVI ~ currently unviable**

* HDD computation time: 7.25 hours to process all 3,913 ga naip tiles
 
## Classification 

**Random Forests**

The random forests classification function streamlines the process of
converting NAIP imagery and it's respective training data to be used with
scikit-learn's random forest classifier.  

Contains `n_jobs` parameter allowing for parallel processing across the CPU
making it ideal due to faster times.
 
Thresholds for the number of estimators in addition to computational time
thresholds will be created. 

Single tile times:
- ARVI w/ 25 estimators ~ 1.9 minutes
- ARVI w/ 100 estimators ~ 5.3 minutes
- ARVI w/ 500 estimators ~ 29.2 minutes - not viable

- nVARI w/ 25 estimators ~ 2.9 minutes
- nVARI w/ 100 estimators ~ 10 minutes

- GRVI w/ 25 estimators - deemed not viable compared to nVARI

TODO: Create RF threshold for time & accuracy

TODO: Solve nVARI classification noise / water body inaccuracy

**Extra Trees Classifier** 

Extra Trees Classifier is considerably quicker/lighter than normal random
 forests but with higher bias. 
 
The parameters that seem to work best: 
```
ExtraTreesClassifier(n_estimators=50, n_jobs=-1,
                     max_features='sqrt', min_samples_leaf=250,
                     class_weight='balanced')
```


