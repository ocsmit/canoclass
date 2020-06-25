###############################################################################
# Canoclass.py
###############################################################################

from osgeo import gdal, ogr
import numpy as np
from scipy import ndimage
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split, cross_val_score


def extra_trees_class(training_raster, training_fit_raster, in_raster,
                      out_tiff, smoothing=True, class_parameters=None):
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
    t = y_raster.GetRasterBand(1).ReadAsArray().astype(np.float64)
    x_raster = gdal.Open(training_fit_raster)
    n = x_raster.GetRasterBand(1).ReadAsArray().astype(np.float64)
    n[np.isnan(n)] = 0
    n_mask = np.ma.MaskedArray(n, mask=(n == 0))
    n_mask.reshape(n.shape)
    y = t[t > 0]
    X = n_mask[t > 0]
    X = X.reshape(-1, 1)

    if class_parameters is None:
        parameters = {"n_estimators": 100, "criterion": 'gini', "max_depth": None,
                             "min_samples_split": 2, "min_samples_leaf": 1,
                             "min_weight_fraction_leaf": 0.0, "max_features": 'auto',
                             "max_leaf_nodes": None, "min_impurity_decrease": 0.0,
                             "min_impurity_split": None, "bootstrap": False,
                             "oob_score": False, "n_jobs": None, "random_state": None,
                             "verbose": 0, "warm_start": False, "class_weight": None,
                             "ccp_alpha": 0.0, "max_samples": None}
        clf = ExtraTreesClassifier(**parameters)
    else:
        parameters = class_parameters
        clf = ExtraTreesClassifier(**parameters)

    ras = clf.fit(X, y)
    r = gdal.Open(in_raster)
    class_raster = r.GetRasterBand(1).ReadAsArray().astype(np.float64)
    class_raster[np.isnan(class_raster)] = 0
    class_mask = np.ma.MaskedArray(class_raster, mask=(class_raster == 0))
    class_mask.reshape(class_raster.shape)
    class_array = class_mask.reshape(-1, 1)

    ras_pre = ras.predict(class_array)
    ras_final = ras_pre.reshape(class_raster.shape)
    ras_byte = ras_final.astype(dtype=np.byte)
    if smoothing:
        smooth_ras = ndimage.median_filter(ras_byte, size=3)
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