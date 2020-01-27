import os
import numpy as np
from osgeo import gdal
from glob import glob

# TODO: start with field creation in naip qq shapefile

def save_raster(in_array, out, dType, snap):
    """

    :param dType:
    :param in_array:
    :param out:
    :param snap:
    :return:
    """

    driver = gdal.GetDriverByName('GTiff')
    metadata = driver.GetMetadata()
    shape = in_array.shape
    dst_ds = driver.Create(out,
                           xsize=shape[1],
                           ysize=shape[0],
                           bands=1,
                           eType=dType)
    proj = snap.GetProjection()
    geo = snap.GetGeoTransform()
    dst_ds.SetGeoTransform(geo)
    dst_ds.SetProjection(proj)
    dst_ds.GetRasterBand(1).WriteArray(in_array)
    dst_ds.FlushCache()
    dst_ds = None

    return in_array


def ARVI(naip_dir, arvi_out):
    # Create list with file names
    for dir, subdir, files in os.walk(naip_dir)

    # Open with gdal & create numpy arrays
    gdal.UseExceptions()
    gdal.AllRegister()
    np.seterr(divide='ignore', invalid='ignore')
    NIR_path = gdal.Open(os.path.join(naip_dir, nir[0]))
    nir_band = NIR_path.GetRasterBand(1).ReadAsArray().astype(np.float32)
    red_path = gdal.Open(os.path.join(naip_dir, red[0]))
    red_band = red_path.GetRasterBand(1).ReadAsArray().astype(np.float32)
    blue_path = gdal.Open(os.path.join(naip_dir, blue[0]))
    blue_band = blue_path.GetRasterBand(1).ReadAsArray().astype(np.float32)
    snap = gdal.Open(os.path.join(naip_dir, red[0]))

    # Perform Calculation
    arvi = ((nir_band - (2 * red_band) + blue_band) / (nir_band + (2 * red_band) + blue_band))

    # Save Raster
    if os.path.exists(arvi_out):
        raise IOError('ARVI raster already created')
    if not os.path.exists(arvi_out):
        save_raster(arvi, arvi_out, gdal.GDT_Float32, snap)

    return arvi, print('Finished')


def VARI(naip_dir, vari_out):

    # Create list with file names
    blue = glob(naip_dir + "/*B2.tif")
    green = glob(naip_dir + "/*B3.tif")
    red = glob(naip_dir + "/*B4.tif")

    # Open with gdal & create numpy arrays
    gdal.UseExceptions()
    gdal.AllRegister()
    np.seterr(divide='ignore', invalid='ignore')
    blue_path = gdal.Open(os.path.join(naip_dir, blue[0]))
    blue_band = blue_path.GetRasterBand(1).ReadAsArray().astype(np.float32)
    green_path = gdal.Open(os.path.join(naip_dir, green[0]))
    green_band = green_path.GetRasterBand(1).ReadAsArray().astype(np.float32)
    red_path = gdal.Open(os.path.join(naip_dir, red[0]))
    red_band = red_path.GetRasterBand(1).ReadAsArray().astype(np.float32)
    snap = gdal.Open(os.path.join(naip_dir, red[0]))

    # Perform Calculation
    vari = ((green_band - red_band) / (green_band + red_band - blue_band))

    # Save raster
    if os.path.exists(vari_out):
        raise IOError('VARI raster already created')
    if not os.path.exists(vari_out):
        save_raster(vari, vari_out, gdal.GDT_Float32, snap)

    return vari, print('Finished')