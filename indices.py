import os
import numpy as np
from osgeo import gdal


def ARVI(naip_dir, out_dir):
    # Create list with file names
    gdal.PushErrorHandler('CPLQuietErrorHandler')
    gdal.UseExceptions()
    gdal.AllRegister()
    np.seterr(divide='ignore', invalid='ignore')

    for dir, subdir, files in os.walk(naip_dir):
        for f in files:
            if f.endswith('.tif'):
                # Open with gdal & create numpy arrays
                naip = gdal.Open(os.path.join(dir, f))
                red_band = naip.GetRasterBand(1).ReadAsArray().astype(np.float32)
                blue_band = naip.GetRasterBand(3).ReadAsArray().astype(np.float32)
                nir_band = naip.GetRasterBand(4).ReadAsArray().astype(np.float32)
                snap = naip

                # Perform Calculation
                arvi = ((nir_band - (2 * red_band) + blue_band) / (nir_band + (2 * red_band) + blue_band))
                name = 'arvi_' + str(f)
                # Save Raster
                if os.path.exists(os.path.join(out_dir, name)):
                    print('Already exists')
                    break
                if not os.path.exists(os.path.join(out_dir, name)):
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
    print('Finished')


def VARI(naip_dir, out_dir):
    for dir, subdir, files in os.walk(naip_dir):
        for f in files:
            if f.endswith('.tif'):
                # Open with gdal & create numpy arrays
                naip = gdal.Open(os.path.join(dir, f))
                red_band = naip.GetRasterBand(1).ReadAsArray().astype(np.float32)
                green_band = naip.GetRasterBand(2).ReadAsArray().astype(np.float32)
                blue_band = naip.GetRasterBand(3).ReadAsArray().astype(np.float32)
                snap = naip

                # Perform Calculation
                vari = ((green_band - red_band) / (green_band + red_band - blue_band))
                name = 'vari_' + str(f)
                # Save Raster
                if os.path.exists(os.path.join(out_dir, name)):
                    print('Already exists')
                    break
                if not os.path.exists(os.path.join(out_dir, name)):
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
    print('Finished')