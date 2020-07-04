from osgeo import gdal
import numpy as np
from scipy import ndimage
from sklearn.ensemble import RandomForestClassifier
from canoclass.load_data import load_data


def rf_class(training_raster, training_fit_raster, in_raster,
             out_tiff, smoothing=True, class_parameters=None):
    """
    This function enables canopy classification of remotely sensed imagery using
    Scikit-learns Random Forests supervised classification algorithm.
    ---
    Args:
        training_raster: Rasterized training data
        training_fit_raster: Raster which data is drawn over
        in_raster: Raster training raster will be applied to
        out_tiff: Final output classified raster
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

    X, y = load_data(training_raster, training_fit_raster)

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
