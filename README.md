# **CanoClass** 

[CanoClass page](https://gislab.isnew.info/open_source_canopy_classification)

## Overview

`CanoClass` is a python module created to process large amounts of NAIP
imagery and create accurate canopy classifications in an open source
framework. Need for an open source classification system arose during the
creation of the Georgia canopy dataset as tools that were being used
, ArcMap and Textron's Feature Analyst, will be phased out within the next
few years. Additionally need for open source arose out of the lack of
insight to the algorithms that were being used by the software to
process our data and no true method to tweak it to suit our needs.

At its core CanoClass is optimized to to solve canopy classification problems.
It is designed to be data agnostic with batch processing functions created to work with NAIP imagery, as scalable processing for NAIP imagery is necessary. 



### Dependencies

- GDAL 
- NumPy
- Scikit-learn
- Rindcalc

### Example

![NAIP_CANOCLASS](https://user-images.githubusercontent.com/55674113/88116578-d8b4b880-cb86-11ea-8a3b-7dd43bf5a0d0.png) 

![ET_CANOCLASS](https://user-images.githubusercontent.com/55674113/88116531-be7ada80-cb86-11ea-85fb-a2c9777142a7.png)


