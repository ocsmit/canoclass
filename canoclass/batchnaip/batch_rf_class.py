import os
from osgeo import gdal, ogr
import numpy as np
from canoclass.batchnaip import config
from scipy import ndimage
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from rindcalc import naip
from canoclass.utils import load_data

def batch_rf_class(pid, smoothing=True, class_parameters=None):
    """
    This function enables batch classification of NAIP imagery using a
    sklearn Extra Trees supervised classification algorithm.
    ---
    Args:
        phy_id: int ::  Physio Id for the region to be processed.
        smoothing: True :: applies median filter to output classified raster
    Keyword Args
        class_parameters: Dict:: arguments for Scikit-learns ET Classifier
            {"n_estimators": 100, "criterion": 'gini', "max_depth": None,
             "min_samples_split": 2, "min_samples_leaf": 1,
             "min_weight_fraction_leaf": 0.0, "max_features": 'auto',
             "max_leaf_nodes": None, "min_impurity_decrease": 0.0,
             "min_impurity_split": None, "bootstrap": True,
             "oob_score": False, "n_jobs": None, "random_state": None,
             "verbose": 0, "warm_start": False, "class_weight": None,
             "ccp_alpha": 0.0, "max_samples": None}

    """

    shp = config.naipqq_shp
    results_dir = config.results
    training_raster = config.training_raster
    training_fit_raster = config.training_fit_raster
    id_field = config.procid_field

    # Query region name, create input and output folder paths
    region_dir = '%s/%s' % (results_dir, pid)
    in_dir = '%s/Inputs' % region_dir
    out_dir = '%s/Outputs' % region_dir
    if not os.path.exists(in_dir):
        raise IOError('Input directory does not exist.')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # Read training & fit raster file and shape to be trained
    X, y = load_data(training_raster, training_fit_raster)

    # Train Random Forests Classifier
    if class_parameters is None:
        parameters = {"n_estimators": 100, "criterion": 'gini', "max_depth": None,
                      "min_samples_split": 2, "min_samples_leaf": 1,
                      "min_weight_fraction_leaf": 0.0, "max_features": 'auto',
                      "max_leaf_nodes": None, "min_impurity_decrease": 0.0,
                      "min_impurity_split": None, "bootstrap": True,
                      "oob_score": False, "n_jobs": None, "random_state": None,
                      "verbose": 0, "warm_start": False, "class_weight": None,
                      "ccp_alpha": 0.0, "max_samples": None}
        clf = RandomForestClassifier(**parameters)
    else:
        parameters = class_parameters
        clf = RandomForestClassifier(**parameters)

    ras = clf.fit(X, y)

    # Open naip_qq shapefile and iterate over attributes to select naip tiles
    # in desired pid.
    src = ogr.Open(shp)
    lyr = src.GetLayer()
    FileName = []
    phyregs = []
    filtered = []
    paths = []
    query = '%d' % pid
    outputs = []
    for i in lyr:
        FileName.append(i.GetField('FileName'))
        phyregs.append(str(i.GetField(id_field)))
    # Get raw file names from naip_qq layer by iterating over phyregs list and
    # retreving corresponding file name from filenames list.
    for j in range(len(phyregs)):
        if query == phyregs[j]:
            filtered.append(FileName[j])
    for i in range(len(filtered)):
        # Edit filenames to get true file names
        # create output filenames and
        # paths.
        file = '%s%s' % ('arvi_', filtered[i])
        filename = '%s.tif' % file[:-13]
        in_path = '%s/%s' % (in_dir, filename)
        out_file = '%s/%s%s' % (out_dir, 'c_', filename)
        outputs.append(out_file)
        paths.append(in_path)
        if os.path.exists(out_file):
            continue
        # Check if input file exists
        if not os.path.exists(paths[i]):
            print('Missing file: ', paths[i])
            continue
        if os.path.exists(paths[i]):
            # If input file exists open with gdal and convert to NumPy array.
            r = gdal.Open(paths[i])
            class_raster = r.GetRasterBand(1).ReadAsArray().astype(
                np.float32)
            class_array = class_raster.reshape(-1, 1)
            # Apply classification
            ras_pre = ras.predict(class_array)
            # Convert back to original shape and make data type Byte
            ras_final = ras_pre.reshape(class_raster.shape)
            ras_byte = ras_final.astype(dtype=np.byte)
            if smoothing:
                # If smoothing = True, apply SciPy median_filter to array and
                # then save.
                smooth_ras = ndimage.median_filter(ras_byte, size=5)
                driver = gdal.GetDriverByName('GTiff')
                metadata = driver.GetMetadata()
                shape = class_raster.shape
                dst_ds = driver.Create(outputs[i],
                                       shape[1],
                                       shape[0],
                                       1,
                                       gdal.GDT_Byte, ['NBITS=2'])
                proj = r.GetProjection()
                geo = r.GetGeoTransform()
                dst_ds.SetGeoTransform(geo)
                dst_ds.SetProjection(proj)
                dst_ds.GetRasterBand(1).WriteArray(smooth_ras)
                dst_ds.FlushCache()
                dst_ds = None
            if not smoothing:
                # If smoothing = False, save numpy array as raster with out
                # smoothing
                driver = gdal.GetDriverByName('GTiff')
                metadata = driver.GetMetadata()
                shape = class_raster.shape
                dst_ds = driver.Create(outputs[i],
                                       shape[1],
                                       shape[0],
                                       1,
                                       gdal.GDT_Byte, ['NBITS=2'])
                proj = r.GetProjection()
                geo = r.GetGeoTransform()
                dst_ds.SetGeoTransform(geo)
                dst_ds.SetProjection(proj)
                dst_ds.GetRasterBand(1).WriteArray(ras_byte)
                dst_ds.FlushCache()
                dst_ds = None

