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

import os
from osgeo import gdal, ogr
import numpy as np

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


def VDVI(naip_dir, out_dir):
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
            name = 'vdvi_%s' % f
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
                    blue_band = naip.GetRasterBand(3).ReadAsArray(). \
                        astype(np.float32)
                    snap = naip

                    a = (2 * green_band - (red_band + blue_band))
                    b = (2 * green_band + (red_band + blue_band))
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

