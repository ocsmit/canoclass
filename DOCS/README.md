# **canopy_foss** 

[CanoPy FOSS page](https://gislab.isnew.info/open_source_canopy_classification)

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

## Dependencies

- GDAL 
- NumPy
- Scikit-learn

## Index functions

**ARVI: ~ NIR based index | Main Focus** 
 
## Classification 

**Extra Trees Classifier** 

## Full Process

  1. Created ARVI
  2. Create Training Data and covert to raster
  3. Classify ARVI
  4. Reproject classified tiles
  5. Clip reproject classified tiles to NAIP QQ seamlines
  6. Mosaic all tiles to region 
  7. Clip mosacied raster to region outline
  
  **Full data creation process except traing data and optimization run with 
  ``create_canopy_dataset``**

For technical information of each function see [TECHNICAL](TECHNICAL.md)
