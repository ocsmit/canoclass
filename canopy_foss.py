################################################################################
# Name:    canopy_foss.py
# Purpose: This module is designed to enable open source based canopy
#          classification utilizing USDA NAIP imagery
# Authors: Owen Smith, IESA, University of North Georgia
# Since:   January, 1st, 2020
################################################################################

import os
from osgeo import gdal, ogr, osr
import numpy as np
import json
import sklearn
from math import ceil
import geopandas as gpd
import canopy_foss.canopy_config_foss as cfg


def create_join_process_naipqq_areas():
    '''
    This function creates a grid based on the extent of the input naip qq
    shapefile and creates a new shapefile with the naip qq grid containing
    filenames and process area ID's

    The size of each grid is measured in degrees.

    Parameters
    naip_qq = file path for naip qq shapefile
    process_areas = file path for out put process area grid
    grid_size_deg = size of each square in degrees
    output_joined_file = file path for final joined output
    '''

    naip_qq_shp = cfg.naipqq_layer
    grid_size_deg = cfg.process_area_degree_size
    process_areas = cfg.process_area_fishnet
    output_joined_file = cfg.process_area_naipqq

    # Get a Layer's Extent
    inShapefile = naip_qq_shp
    inDriver = ogr.GetDriverByName("ESRI Shapefile")
    inDataSource = inDriver.Open(inShapefile, 0)
    inLayer = inDataSource.GetLayer()
    srs = inLayer.GetSpatialRef()
    extent = inLayer.GetExtent()

    # convert sys.argv to float
    xmin = float(extent[0])
    xmax = float(extent[1])
    ymin = float(extent[2])
    ymax = float(extent[3])
    gridWidth = float((grid_size_deg / 2) * 0.0625)
    gridHeight = float((grid_size_deg / 2) * 0.0625)

    # get rows
    rows = ceil((ymax - ymin) / gridHeight)
    # get columns
    cols = ceil((xmax - xmin) / gridWidth)

    # start grid cell envelope
    ringXleftOrigin = xmin
    ringXrightOrigin = xmin + gridWidth
    ringYtopOrigin = ymax
    ringYbottomOrigin = ymax - gridHeight

    # create output file
    outDriver = ogr.GetDriverByName('ESRI Shapefile')
    if os.path.exists(process_areas):
        os.remove(process_areas)
    outDataSource = outDriver.CreateDataSource(process_areas)
    outLayer = outDataSource.CreateLayer(process_areas, srs,
                                         geom_type=ogr.wkbPolygon)
    featureDefn = outLayer.GetLayerDefn()

    # create grid cells
    countcols = 0
    while countcols < cols:
        countcols += 1

        # reset envelope for rows
        ringYtop = ringYtopOrigin
        ringYbottom = ringYbottomOrigin
        countrows = 0

        while countrows < rows:
            countrows += 1
            ring = ogr.Geometry(ogr.wkbLinearRing)
            ring.AddPoint(ringXleftOrigin, ringYtop)
            ring.AddPoint(ringXrightOrigin, ringYtop)
            ring.AddPoint(ringXrightOrigin, ringYbottom)
            ring.AddPoint(ringXleftOrigin, ringYbottom)
            ring.AddPoint(ringXleftOrigin, ringYtop)
            poly = ogr.Geometry(ogr.wkbPolygon)
            poly.AddGeometry(ring)

            # add new geom to layer
            outFeature = ogr.Feature(featureDefn)
            outFeature.SetGeometry(poly)
            outLayer.CreateFeature(outFeature)
            outFeature = None

            # new envelope for next poly
            ringYtop = ringYtop - gridHeight
            ringYbottom = ringYbottom - gridHeight

        # new envelope for next poly
        ringXleftOrigin = ringXleftOrigin + gridWidth
        ringXrightOrigin = ringXrightOrigin + gridWidth

    # Save and close DataSources
    outDataSource = None

    naip = gpd.read_file(naip_qq_shp)
    areas = gpd.read_file(process_areas)

    joined_naip = gpd.sjoin(naip, areas, how='inner', op='intersects')
    joined_naip = joined_naip.rename(columns={'FID': 'PROCESS_ID'})
    joined_naip = joined_naip[['FileName', 'PROCESS_ID', 'geometry']]
    if os.path.exists(output_joined_file):
        os.remove(output_joined_file)
    joined_naip.to_file(output_joined_file)

    print('Finished')

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
                    red_band = naip.GetRasterBand(1).ReadAsArray()\
                        .astype(np.float32)
                    blue_band = naip.GetRasterBand(3).ReadAsArray()\
                        .astype(np.float32)
                    nir_band = naip.GetRasterBand(4).ReadAsArray()\
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

def prepare_training_data():

    return

def support_vector_class():

    return

def linear_reg_class():

    return
