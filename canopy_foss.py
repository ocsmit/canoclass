# ------------------------------------------------------------------------------
# Name:    canopy_foss.py
# Purpose: This module is designed to enable open source based canopy
#          classification utilizing USDA NAIP imagery
# Authors: Owen Smith, IESA, University of North Georgia
# Since:   January, 1st, 2020
# ------------------------------------------------------------------------------

import os
from osgeo import gdal, ogr
import numpy as np
from sklearn import linear_model


# import canopy_foss.canopy_config_foss as cfg

# ==============================================================================
# Preprocessing Functions:
# ------------------------
#       Index Calculations:
#       -- ARVI(naip_dir, out_dir)
#       -- VARI(naip_dir, out_dir)
#       -- GRVI(naip_dir, out_dir)
#
#       Training Data Prep:
#       -- prepare_training_data(vector, ref_raster, out_raster, field='id')
# ==============================================================================

def ARVI(naip_dir, out_dir):
    """
    This function walks through the input NAIP directory and performs the
    ARVI calculation on each naip geotiff file and saves each new ARVI
    geotiff in the output directory with the prefix 'arvi_'
    ---
    Args:
        naip_dir: Folder which contains all subfolders of naip imagery
        out_dir:  Folder in which all calculated geotiff's are saved
    """
    # naip_dir = cfg.naip_path
    # out_dir = cfg.index_out_path % 'ARVI'

    if not os.path.exists(naip_dir):
        print('NAIP directory not found')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # Create list with file names
    gdal.PushErrorHandler('CPLQuietErrorHandler')
    gdal.UseExceptions()
    gdal.AllRegister()
    np.seterr(divide='ignore', invalid='ignore')

    for dir, subdir, files in os.walk(naip_dir):
        for f in files:
            name = 'arvi_%s' % f
            if os.path.exists(os.path.join(out_dir, name)):
                continue
            if not os.path.exists(os.path.join(out_dir, name)):
                if f.endswith('.tif'):
                    # Open with gdal & create numpy arrays
                    naip = gdal.Open(os.path.join(dir, f))
                    red_band = naip.GetRasterBand(1).ReadAsArray() \
                        .astype(np.float32)
                    blue_band = naip.GetRasterBand(3).ReadAsArray() \
                        .astype(np.float32)
                    nir_band = naip.GetRasterBand(4).ReadAsArray() \
                        .astype(np.float32)
                    snap = naip

                    # Perform Calculation
                    a = (nir_band - (2 * red_band) + blue_band)
                    b = (nir_band + (2 * red_band) + blue_band)
                    arvi = a / b
                    name = 'arvi_' + str(f)
                    # Save Raster

                    driver = gdal.GetDriverByName('GTiff')
                    metadata = driver.GetMetadata()
                    shape = arvi.shape
                    dst_ds = driver.Create(os.path.join(out_dir, name),
                                           xsize=shape[1],
                                           ysize=shape[0],
                                           bands=1,
                                           eType=gdal.GDT_Float32)
                    proj = snap.GetProjection()
                    geo = snap.GetGeoTransform()
                    dst_ds.SetGeoTransform(geo)
                    dst_ds.SetProjection(proj)
                    dst_ds.GetRasterBand(1).WriteArray(arvi)
                    dst_ds.FlushCache()
                    dst_ds = None
                    print(name)

    print('Finished')


def nVARI(naip_dir, out_dir):
    """
    This function walks through the input NAIP directory and performs the
    VARI calculation on each naip geotiff file and saves each new VARI
    geotiff in the output directory with the prefix 'arvi_'
    ---
    Args:
        naip_dir: Folder which contains all subfolders of naip imagery
        out_dir: Folder in which all calculated geotiff's are saved
    """
    # naip_dir = cfg.naip_path
    # out_dir = cfg.index_out_path % 'VARI'

    if not os.path.exists(naip_dir):
        print('NAIP directory not found')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    gdal.PushErrorHandler('CPLQuietErrorHandler')
    gdal.UseExceptions()
    gdal.AllRegister()
    np.seterr(divide='ignore', invalid='ignore')

    def norm(array):
        array_min, array_max = array.min(), array.max()
        return ((1 - 0) * ((array - array_min) / (array_max - array_min))) + 1

    for dir, subdir, files in os.walk(naip_dir):
        for f in files:
            name = 'vari_%s' % f
            if os.path.exists(os.path.join(out_dir, name)):
                continue
            if not os.path.exists(os.path.join(out_dir, name)):
                if f.endswith('.tif'):
                    # Open with gdal & create numpy arrays
                    naip = gdal.Open(os.path.join(dir, f))
                    red_band = norm(naip.GetRasterBand(1).ReadAsArray().
                                    astype(np.float32))
                    green_band = norm(naip.GetRasterBand(2).ReadAsArray().
                                      astype(np.float32))
                    blue_band = norm(naip.GetRasterBand(3).ReadAsArray().
                                     astype(np.float32))
                    snap = naip

                    a = (green_band - red_band)
                    b = (green_band + red_band - blue_band)
                    # Perform Calculation
                    vari = a / b

                    # Save Raster
                    driver = gdal.GetDriverByName('GTiff')
                    metadata = driver.GetMetadata()
                    shape = vari.shape
                    dst_ds = driver.Create(os.path.join(out_dir, name),
                                           xsize=shape[1],
                                           ysize=shape[0],
                                           bands=1,
                                           eType=gdal.GDT_Float32)
                    proj = snap.GetProjection()
                    geo = snap.GetGeoTransform()
                    dst_ds.SetGeoTransform(geo)
                    dst_ds.SetProjection(proj)
                    dst_ds.GetRasterBand(1).WriteArray(vari)
                    dst_ds.FlushCache()
                    dst_ds = None
                    print(name)

    print('Finished')


def GRVI(naip_dir, out_dir):
    """
    This function walks through the input NAIP directory and performs the
    GRVI calculation on each naip geotiff file and saves each new GRVI
    geotiff in the output directory with the prefix 'grvi_'
    ---
    Args:
        naip_dir: Folder which contains all subfolders of naip imagery
        out_dir: Folder in which all calculated geotiff's are saved
    """

    if not os.path.exists(naip_dir):
        print('NAIP directory not found')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    gdal.PushErrorHandler('CPLQuietErrorHandler')
    gdal.UseExceptions()
    gdal.AllRegister()
    np.seterr(divide='ignore', invalid='ignore')

    for dir, subdir, files in os.walk(naip_dir):
        for f in files:
            name = 'grvi_%s' % f
            if os.path.exists(os.path.join(out_dir, name)):
                continue
            if not os.path.exists(os.path.join(out_dir, name)):
                if f.endswith('.tif'):
                    # Open with gdal & create numpy arrays
                    naip = gdal.Open(os.path.join(dir, f))
                    red_band = naip.GetRasterBand(1).ReadAsArray(). \
                        astype(np.float32)
                    green_band = naip.GetRasterBand(2).ReadAsArray(). \
                        astype(np.float32)
                    snap = naip

                    a = (green_band - red_band)
                    b = (green_band + red_band)
                    # Perform Calculation
                    grvi = a / b

                    # Save Raster
                    driver = gdal.GetDriverByName('GTiff')
                    metadata = driver.GetMetadata()
                    shape = grvi.shape
                    dst_ds = driver.Create(os.path.join(out_dir, name),
                                           xsize=shape[1],
                                           ysize=shape[0],
                                           bands=1,
                                           eType=gdal.GDT_Float32)
                    proj = snap.GetProjection()
                    geo = snap.GetGeoTransform()
                    dst_ds.SetGeoTransform(geo)
                    dst_ds.SetProjection(proj)
                    dst_ds.GetRasterBand(1).WriteArray(grvi)
                    dst_ds.FlushCache()
                    dst_ds = None
                    print(name)

    print('Finished')


def prepare_training_data(vector, ref_raster, out_raster, field='id'):
    """
    This function converts the training data shapefile into a raster to allow
    the training data to be applied for classification
    ---
    Args:
        vector:
        ref_raster:
        out_raster:
        field:
    """
    # TODO: Allow for training data to have 0 and 1 as values

    snap = gdal.Open(ref_raster)
    shp = ogr.Open(vector)
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


# ==============================================================================
# Classification Functions:
# -------------------------
#       Random Forests:
#       -- random_forests_class(training_raster, in_raster, out_tiff)
# ==============================================================================


def random_forests_class(training_raster, training_fit_raster, in_raster,
                         out_tiff):
    """
    This function enables classification of NAIP imagery using a sklearn Random
    Forests supervised classification algorithm.
    ---
    Args:
        training_fit_raster:
        training_raster: Rasterized training data
        in_raster: Raster training raster will be applied to 
        out_tiff: Final output classified raster
    """
    # TODO: Refactor

    from sklearn.ensemble import RandomForestClassifier

    y_raster = gdal.Open(training_raster)
    t = y_raster.GetRasterBand(1).ReadAsArray().astype(np.float32)
    x_raster = gdal.Open(training_fit_raster)
    n = x_raster.GetRasterBand(1).ReadAsArray().astype(np.float32)
    y = t[t > 0]
    X = n[t > 0]
    X = X.reshape(-1, 1)

    clf = RandomForestClassifier(n_estimators=25, n_jobs=-1)
    ras = clf.fit(X, y)

    r = gdal.Open(in_raster)
    class_raster = r.GetRasterBand(1).ReadAsArray().astype(np.float32)
    class_array = class_raster.reshape(-1, 1)
    ras_pre = ras.predict(class_array)
    ras_final = ras_pre.reshape(class_raster.shape)
    ras_byte = ras_final.astype(dtype=np.byte)

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

    print('Classified raster complete.')


def linear_reg(training_raster, naip):
    """
    This module performs linear regression analysis on naip data
    to classify canopy
    """

    x_raster = gdal.Open(training_raster)
    X = x_raster.GetRasterBand(1).ReadAsArray().astype(np.float32)

    y_raster = gdal.Open(naip)
    y = y_raster.GetRasterBand(1).ReadAsArray().astype(np.float32)

    reg = linear_model.LinearRegression(n_jobs=-1).fit(X=X, y=y)

    return reg
