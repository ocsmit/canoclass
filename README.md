# canopy_foss

Open source canopy and deforestation monitoring.

## Overview

canopy_foss is a python module created to process large amounts of NAIP imagery and create accurate canopy classificatons in an open source framework. Need for an open source classification system arose during the creation of the Georgia canopy dataset as tools that were being used, ArcMap and Textron's Feature Analyst, will be phased out within the next few years. Additionally need for open source arose out of the lack of insight to the algorthims that were being used by the software to process our data and no true method to tweak it to suit our needs.

## Index functions

All index functions are stored in `canopy_foss.indicies.py`

**ARVI** 

* SSD computation time: 1.45 seconds per calculation or approx. 1:35 hours for full computation of all 3,913 ga naip tiles to NVME SSD. 

* HDD computation time: 6.16 hours hours to proccess all 3,913 ga naip tiles

**VARI:** computation speed still needs to be tested


