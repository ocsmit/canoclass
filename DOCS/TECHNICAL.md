**ARVI: ~ NIR based index | Main Focus** 

* SSD computation time: 1.45 seconds per calculation or approx. 1:35 hours 
  for full computation of all 3,913 ga naip tiles to NVME SSD. 

* HDD computation time: 6.16 hours hours to process all 3,913 ga naip tiles

**nVARI:**

* HDD computation time: 5:45 hours to process all 3,913 ga naip tiles

* nVARI bands are normalized between -1 and 1 before being computed as the VARI
  formula [(green - red) / (green + red - blue)] ensures the values do not
  inherently fall between -1 and 1. Normalization ensures consistency for data
  across the entire NAIP data set. 

**Random Forests**

**Training data score = 0.8947517085091881**

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

TODO: Create RF threshold for time & accuracy

TODO: Solve nVARI classification noise / water body inaccuracy

**Extra Trees Classifier ~ Main focus** 

**Training data score = 0.8947517085091881**

Extra Trees Classifier is considerably quicker/lighter than normal random
 forests but with higher bias. Lighter computation load, more random. 
 
Single tile times:
- ARVI w/ 50 estimators ~ 1 minute
 
Current Parameters
```
    ExtraTreesClassifier(n_estimators=50, n_jobs=-1,
                         max_features='sqrt', min_samples_leaf=10)
```
