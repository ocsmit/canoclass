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
- Rindcalc

## Proccess & Setting up the NAIP QQ shapefile
All functions created for the process work by reading filenames and
 physiographic ids from the NAIP QQ shapefile. However, the NAIP QQ file does 
 not contain the physio id’s without additional processing. To enable each 
 region to be queried within the NAIP QQ shapefile the “Join Attributes by 
 Location” tool within QGIS 3.10 is used to join the physiographic district 
 shapefile attributes to the NAIP QQ shapefile. The join performed is a ‘one 
 for many join’ for all NAIP QQ’s either within or intersecting a physiographic 
 district. The ‘one for many join’ is important as one NAIP QQ could potentially
be within multiple districts, so a new polygon feature needs to be created
with each physio id. The spatial join method will allow for the physiographic
districts and their corresponding files to be queried directly using OGR, 
GDAL’s vector processing API. The joined NAIP QQ is saved as a new shapefile.

The original unjoined NAIP QQ shapefile is still used however, as GDAL is not
 able to properly read the required geometry required for clipping of each QQ
 feature due to the ‘one for many join’. The original shapefile will
 subsequently be used for clipping the tiles their boundaries after the file
  names are queried from the joined NAIP QQ shapefile. 


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
