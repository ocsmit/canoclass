# ==============================================================================
# Classification Functions:
# -------------------------
#       -- split_data(training_raster, training_fit_raster)
#       -- tune_hyperparameter(training_raster, training_fit_raster)
#       Random Forests:
#       -- random_forests_class(training_raster, training_fit_raster, in_raster,
#                               out_tiff, smoothing=True)
#       -- batch_random_forests(in_directory, training_raster, fit_raster,
#                               out_directory, smoothing=True)
#       Extra Trees:
#       -- extra_trees_class(training_raster, training_fit_raster, in_raster,
#                            out_tiff, smoothing=True):
#       -- batch_extra_trees(in_directory, training_raster, fit_raster,
#                            out_directory, smoothing=True):
# ==============================================================================

import os
from osgeo import gdal, ogr
import numpy as np
from scipy import ndimage
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split

def split_data(training_raster, training_fit_raster):

    y_raster = gdal.Open(training_raster)
    t = y_raster.GetRasterBand(1).ReadAsArray().astype(np.float32)
    x_raster = gdal.Open(training_fit_raster)
    n = x_raster.GetRasterBand(1).ReadAsArray().astype(np.float32)
    y = t[t > 0]
    X = n[t > 0]
    X = X.reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =
    0.33)

    return X_train, X_test, y_train, y_test


# TODO: WIP
def tune_hyperparameter(training_raster, training_fit_raster):

    y_raster = gdal.Open(training_raster)
    t = y_raster.GetRasterBand(1).ReadAsArray().astype(np.float32)
    x_raster = gdal.Open(training_fit_raster)
    n = x_raster.GetRasterBand(1).ReadAsArray().astype(np.float32)
    y = t[t > 0]
    X = n[t > 0]
    X = X.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =
    0.33)

    n_estimators = [int(x) for x in np.linspace(start=10, stop=150, num=10)]
    min_samples_leaf = [int(x) for x in np.linspace(start=5, stop=500, num=10)]
    random_grid = {
        'n_estimators': n_estimators,
        'min_samples_leaf': min_samples_leaf
    }
    weight = [{1: 1, 2: 2}]
    etc = ExtraTreesClassifier(n_estimators=100, n_jobs=-1,
                               max_features=None,
                               min_samples_leaf=10, class_weight={1: 2, 2: 0.5})
    clf = RandomizedSearchCV(etc, random_grid, random_state=0)
    clf.fit(X_train, y_train)

    print(clf.best_params_)



def random_forests_class(training_raster, training_fit_raster, in_raster,
                         out_tiff, smoothing=True):
    """
    This function enables classification of NAIP imagery using a sklearn Random
    Forests supervised classification algorithm.
    ---
    Args:
        training_fit_raster:
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

    clf = RandomForestClassifier(n_estimators=50, n_jobs=2,
                                 max_features='sqrt',
                                 min_samples_leaf=10)
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


def batch_random_forests(in_directory, training_raster, fit_raster,
                         out_directory, smoothing=True):
    """
    This function enables batch classification of NAIP imagery using a
    sklearn Ec supervised classification algorithm.
    ---
    Args:
        in_directory: Input naip directory
        training_raster: Rasterized training data
        fit_raster: Raster training raster will be applied to
        out_directory: output directory for classified imagery
        smoothing: True :: applies median filter to output classified raster
    """
    if not os.path.exists(out_directory):
        os.mkdir(out_directory)
    for dir, subdir, files in os.walk(in_directory):
        for f in files:
            input_raster = os.path.join(in_directory, f)
            output = os.path.join(out_directory, 'erf_' + f)
            if os.path.exists(output):
                continue
            if not os.path.exists(output):
                random_forests_class(training_raster, fit_raster,
                                     input_raster, output, smoothing)
    print('Complete.')


def extra_trees_class(training_raster, training_fit_raster, in_raster,
                      out_tiff, smoothing=True):
    """
    This function enables classification of NAIP imagery using a sklearn Random
    Forests supervised classification algorithm.
    ---
    Args:
        training_fit_raster:
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

    weight = [{1: 1, 2: 2}]
    # TODO: Quantify min_samples_leaf threshold
    clf = ExtraTreesClassifier(n_estimators=100, n_jobs=-1,
                               max_features=None,
                               min_samples_leaf=10, class_weight={1: 2,
                                                                  2: 0.5})
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


def batch_extra_trees(in_directory, training_raster, fit_raster, out_directory,
                      smoothing=True):
    """
    This function enables batch classification of NAIP imagery using a
    sklearn Extra Trees supervised classification algorithm.
    ---
    Args:
        in_directory: Input naip directory
        training_raster: Rasterized training data
        fit_raster: Raster training raster will be applied to
        out_directory: output directory for classified imagery
        smoothing: True :: applies median filter to output classified raster
    """
    if not os.path.exists(out_directory):
        os.mkdir(out_directory)
    for dir, subdir, files in os.walk(in_directory):
        for f in files:
            input_raster = os.path.join(in_directory, f)
            output = os.path.join(out_directory, 'rf_' + f)
            if os.path.exists(output):
                continue
            if not os.path.exists(output):
                extra_trees_class(training_raster, fit_raster,
                                  input_raster, output, smoothing)
    print('Complete.')
