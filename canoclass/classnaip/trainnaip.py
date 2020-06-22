# ==============================================================================
# Title: training.py
# Author: Owen Smith, University of North Georgia
# Preprocessing Functions:
# ------------------------
#       Index Calculation:
#       -- ARVI(input_naip, out_naip)
#
#       Training Data Prep:
#       -- prepare_training_data(vector, ref_raster, out_raster, field='id')
#       -- split_data(training_raster, training_fit_raster)
#       -- tune_hyperparameter(training_raster, training_fit_raster)
#       -- extra_trees_class(training_raster, training_fit_raster, in_raster,
#                            out_tiff, smoothing=True)
# ==============================================================================

import os
from osgeo import gdal, ogr
import numpy as np
from scipy import ndimage
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split, cross_val_score
from rindcalc import naip


def veg_index(input_naip, out_naip, index='ARVI'):
    """
    Wrapper function to create the desired vegetation index for training using
    rindcalc.
    Args:
        input_naip: Input NAIP tile
        out_naip: Output ARVI index raster
        index: Which vegatation index to create
    """
    if not os.path.exists(input_naip):
        raise IOError('Path not found.')
    if os.path.exists(input_naip):
        i = getattr(naip, index)(input_naip, out_naip)

    print('Finished')


def prepare_training_data(input_shp, reference_raster, out_raster, field='id'):
    """
    This function converts the training data shapefile into a raster to allow
    the training data to be applied for classification
    ---
    Args:
        input_shp: Input shapefile to rasterize
        reference_raster: Raster which shapefile was drawn over
        out_raster: Output training raster
        field: Field to rasterize
    """
    # TODO: Allow for training data to have 0 and 1 as values

    snap = gdal.Open(reference_raster)
    shp = ogr.Open(input_shp)
    layer = shp.GetLayer()

    xy = snap.GetRasterBand(1).ReadAsArray().astype(np.float32).shape

    driver = gdal.GetDriverByName('GTiff')
    metadata = driver.GetMetadata()
    dst_ds = driver.Create(out_raster,
                           xsize=xy[1],
                           ysize=xy[0],
                           bands=1,
                           eType=gdal.GDT_Byte)
    proj = snap.GetProjection()
    geo = snap.GetGeoTransform()
    dst_ds.SetGeoTransform(geo)
    dst_ds.SetProjection(proj)
    if field is None:
        gdal.RasterizeLayer(dst_ds, [1], layer, None)
    else:
        OPTIONS = ['ATTRIBUTE=' + field]
        gdal.RasterizeLayer(dst_ds, [1], layer, None, options=OPTIONS)
    dst_ds.FlushCache()
    dst_ds = None

    print('Vector to raster complete.')
    return out_raster


def split_data(training_raster, training_fit_raster):
    """
    Split data into training and testing data

    Args:
        training_raster: Rasterized training data
        training_fit_raster: Raster which data is drawn over

    """

    y_raster = gdal.Open(training_raster)
    t = y_raster.GetRasterBand(1).ReadAsArray().astype(np.float32)
    x_raster = gdal.Open(training_fit_raster)
    n = x_raster.GetRasterBand(1).ReadAsArray().astype(np.float32)
    y = t[t > 0]
    X = n[t > 0]
    X = X.reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    return X_train, X_test, y_train, y_test


def tune_hyperparameter(training_raster, training_fit_raster):
    """
    Performs 5 fold cross validation to determine optimal parameters

    Args:
        training_raster: Rasterized training data
        training_fit_raster: Raster which data is drawn over

    """

    y_raster = gdal.Open(training_raster)
    t = y_raster.GetRasterBand(1).ReadAsArray().astype(np.float32)
    x_raster = gdal.Open(training_fit_raster)
    n = x_raster.GetRasterBand(1).ReadAsArray().astype(np.float32)
    y = t[t > 0]
    X = n[t > 0]
    X = X.reshape(-1, 1)

    X_train, X_test, y_train, y_test = split_data(training_raster,
                                                  training_fit_raster)

    n_estimators = [int(x) for x in np.linspace(start=10, stop=150, num=10)]
    min_samples_leaf = [int(x) for x in np.linspace(start=5, stop=500, num=10)]
    random_grid = {
        'n_estimators': n_estimators,
        # 'min_samples_leaf': min_samples_leaf
    }
    etc = ExtraTreesClassifier(n_jobs=-1, max_features=None)
    clf = RandomizedSearchCV(etc, random_grid, random_state=0, verbose=3)
    clf.fit(X_train, y_train)

    print(clf.best_params_)
    return clf.cv_results_


def extra_trees_class(training_raster, training_fit_raster, in_raster,
                      out_tiff, smoothing=True):
    """
    This function enables classification of NAIP imagery using a sklearn Random
    Forests supervised classification algorithm.
    ---
    Args:
        training_fit_raster: Raster which data is drawn over
        training_raster: Rasterized training data
        in_raster: Raster training raster will be applied to
        out_tiff: Final output classified raster
        smoothing: True :: applies median filter to output classified raster
    """
    y_raster = gdal.Open(training_raster)
    t = y_raster.GetRasterBand(1).ReadAsArray().astype(np.float32)
    x_raster = gdal.Open(training_fit_raster)
    n = x_raster.GetRasterBand(1).ReadAsArray().astype(np.float32)
    y = t[t > 0]
    X = n[t > 0]
    X = X.reshape(-1, 1)
    clf = ExtraTreesClassifier(n_estimators=100, n_jobs=-1,
                               max_features=None,
                               min_samples_leaf=10, class_weight={1: 2, 2: 0.5})
    scores = cross_val_score(clf, X, y, cv=5)
    print(scores)
    ras = clf.fit(X, y)
    r = gdal.Open(in_raster)
    class_raster = r.GetRasterBand(1).ReadAsArray().astype(np.float32)
    class_array = class_raster.reshape(-1, 1)
    ras_pre = ras.predict(class_array)
    ras_final = ras_pre.reshape(class_raster.shape)
    ras_byte = ras_final.astype(dtype=np.byte)
    if smoothing:
        smooth_ras = ndimage.median_filter(ras_byte, size=5)
        driver = gdal.GetDriverByName('GTiff')
        metadata = driver.GetMetadata()
        shape = class_raster.shape
        dst_ds = driver.Create(out_tiff,
                               xsize=shape[1],
                               ysize=shape[0],
                               bands=1,
                               eType=gdal.GDT_Byte)
        proj = r.GetProjection()
        geo = r.GetGeoTransform()
        dst_ds.SetGeoTransform(geo)
        dst_ds.SetProjection(proj)
        dst_ds.GetRasterBand(1).WriteArray(smooth_ras)
        dst_ds.FlushCache()
        dst_ds = None
    if not smoothing:
        driver = gdal.GetDriverByName('GTiff')
        metadata = driver.GetMetadata()
        shape = class_raster.shape
        dst_ds = driver.Create(out_tiff,
                               xsize=shape[1],
                               ysize=shape[0],
                               bands=1,
                               eType=gdal.GDT_Byte)
        proj = r.GetProjection()
        geo = r.GetGeoTransform()
        dst_ds.SetGeoTransform(geo)
        dst_ds.SetProjection(proj)
        dst_ds.GetRasterBand(1).WriteArray(ras_byte)
        dst_ds.FlushCache()
        dst_ds = None

    print(out_tiff)
