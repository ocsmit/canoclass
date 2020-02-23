# ------------------------------------------------------------------------------
# Name:    canopy_foss.py
# Purpose: This module is designed to enable open source based canopy
#          classification utilizing USDA NAIP imagery
# Authors: Owen Smith, IESA, University of North Georgia
# Since:   January, 1st, 2020
# ------------------------------------------------------------------------------

import os
from osgeo import gdal, ogr, osr
import numpy as np
import json
from sklearn import linear_model
import canopy_foss.canopy_config_foss as cfg


def norm(array):
    array_min, array_max = array.min(), array.max()
    return ((1 - 0) * ((array - array_min) / (array_max - array_min))) + 1


def ARVI():
    """
    This function walks through the input NAIP directory and performs the
    ARVI calculation on each naip geotiff file and saves each new ARVI
    geotiff in the output directory with the prefix 'arvi_'
    ---
    Parameters:
    naip_dir = Folder which contains all subfolders of naip imagery
    out_dir = Folder in which all calculated geotiff's are saved
    """
    naip_dir = cfg.naip_path
    out_dir = cfg.index_out_path % 'ARVI'

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
            name = 'arvi_' + str(f)
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


def VARI():
    """
    This function walks through the input NAIP directory and performs the
    VARI calculation on each naip geotiff file and saves each new VARI
    geotiff in the output directory with the prefix 'arvi_'
    ---
    Parameters:
    naip_dir = Folder which contains all subfolders of naip imagery
    out_dir = Folder in which all calculated geotiff's are saved
    """
    naip_dir = cfg.naip_path
    out_dir = cfg.index_out_path % 'VARI'

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
            name = 'vari_' + str(f)
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
                    name = 'vari_' + str(f)

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


def prepare_training_data(vector, out_raster, field='id'):
    # WIP
    snap_raster = cfg.snaprast_path

    snap = gdal.Open(snap_raster)
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

    return out_raster


def support_vector_class(training_raster, naip):

    x_raster = gdal.Open(training_raster)
    X = x_raster.GetRasterBand(1).ReadAsArray().astype(np.float32)


    y_raster = gdal.Open(naip)
    y = y_raster.GetRasterBand(1).ReadAsArray().astype(np.float32)

    return


def linear_reg(training_raster, naip):
    '''
    This module performs linear regression analysis on naip data
    to classify canopy
    '''

    x_raster = gdal.Open(training_raster)
    X = x_raster.GetRasterBand(1).ReadAsArray().astype(np.float32)

    y_raster = gdal.Open(naip)
    y = y_raster.GetRasterBand(1).ReadAsArray().astype(np.float32)

    reg = linear_model.LinearRegression(n_jobs=-1).fit(X=X, y=y)

    return reg
