# canopy_foss

Open source canopy and deforestation monitoring.

## Overview

canopy_foss is a python module created to process large amounts of NAIP
imagery and create accurate canopy classificatons in an open source
framework. Need for an open source classification system arose during the
creation of the Georgia canopy dataset as tools that were being used
, ArcMap and Textron's Feature Analyst, will be phased out within the next
few years. Additionally need for open source arose out of the lack of
insight to the algorthims that were being used by the software to
process our data and no true method to tweak it to suit our needs.

## Index functions

All index functions are stored in `canopy_foss.indicies.py`

**ARVI:** 

* SSD computation time: 1.45 seconds per calculation or approx. 1:35 hours 
  for full computation of all 3,913 ga naip tiles to NVME SSD. 

* HDD computation time: 6.16 hours hours to proccess all 3,913 ga naip tiles

**VARI:** 

* HDD computation time: 5:45 hours to process all 3,913 ga naip tiles

 
## Classification 

**Random Forests**

Contains n_jobs parameter allowing for parallel processign across the CPU
 making it ideal due to faster times.

Single tile times:
- ARVI w/ 25 estimators ~ 1.9 minutes
- ARVI w/ 100 estimators ~ 5.3 minutes
- ARVI w/ 500 estimators ~ 29.2 minutes - not viable
- VARI w/ 25 estimators ~ 2.9 minutes
- VARI w/ 100 estimators ~ 10 minutes
