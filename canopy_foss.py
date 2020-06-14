# ==============================================================================
# Title: canopy_foss.py
# Author: Owen Smith, University of North Georgia
# Canopy data creation process:
# -----------------------------
#       * View readme to see needed data
#       * Ensure all configuration parameters are set before starting process.
#       * To create training data and determine paramters for Extra Trees
#         Classifier, use training.py which contains all preprocessing function-
#         s.
#
#       1. ARVI(phy_id)
#       2. batch_extra_trees(phy_id, smoothing=True)
#       3. clip_reproject_classified_tiles(phy_id)
#       4. mosaic_tiles(phy_id)
#       5. clip_mosaic(phy_id)
#
#       wrapper function to perform all steps:
#           * create_canopy_dataset(phy_id)
#
# ==============================================================================

import os
from osgeo import gdal, ogr
import numpy as np
import config
from scipy import ndimage
from sklearn.ensemble import ExtraTreesClassifier
from rindcalc import naip

def get_phyregs_name(phy_id):
    shp = config.phyreg_lyr
    src = ogr.Open(shp)
    lyr = src.GetLayer()
    query = "PHYSIO_ID = %d" % phy_id
    lyr.SetAttributeFilter(query)
    for i in lyr:
        name = i.GetField('NAME')
        name = name.replace(' ', '_').replace('-', '_')
    return name


def batch_veg_index(phy_id, index='ARVI'):
    """
    This function walks through the input NAIP directory and performs the
    ARVI calculation on each naip geotiff file and saves each new ARVI
    geotiff in the output directory with the prefix 'arvi_'
    ---
    Args:
        phy_id: int ::  Physio Id for the region to be processed.
    """
    workspace = config.workspace
    shp = config.naipqq_shp
    naip_dir = config.naip_dir
    results_dir = config.results

    if not os.path.exists(naip_dir):
        print('NAIP directory not found')
    # Get region name and create output file path
    region = get_phyregs_name(phy_id)
    print(region)
    region_dir = '%s/%s' % (results_dir, region)
    out_dir = '%s/Inputs' % region_dir
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    if not os.path.exists(out_dir):
        os.mkdir(region_dir)
        os.mkdir(out_dir)
    gdal.PushErrorHandler('CPLQuietErrorHandler')
    gdal.UseExceptions()
    gdal.AllRegister()
    np.seterr(divide='ignore', invalid='ignore')

    # Open naip_qq shapefile and iterate over attributes to select naip tiles
    # in desired phy_id.
    src = ogr.Open(shp)
    lyr = src.GetLayer()
    FileName = []
    phyregs = []
    filtered = []
    paths = []
    query = '%d' % phy_id
    outputs = []
    # Query is done by iterating over list of entire naip_qq shapefile.
    # ogr.SetAttributeFilter throws SQL expression error due to needed commas
    # around phy_id.
    for i in lyr:
        FileName.append(i.GetField('FileName'))
        phyregs.append(str(i.GetField('PHYSIO_ID')))
    # Get raw file names from naip_qq layer by iterating over phyregs list and
    # retreving corresponding file name from filenames list.
    for j in range(len(phyregs)):
        if query == phyregs[j]:
            filtered.append(FileName[j])
    # Edit filenames to get true file names, and create output filenames and
    # paths.
    for i in range(len(filtered)):
        file = filtered[i]
        filename = '%s.tif' % file[:-13]
        arvi_file = 'arvi_%s' % filename
        folder = file[2:7]
        in_path = '%s/%s/%s' % (naip_dir, folder, filename)
        out_path = '%s/%s' % (out_dir, arvi_file)
        outputs.append(out_path)
        paths.append(in_path)
        # If output exists, move to next naip tile.
        if os.path.exists(outputs[i]):
            continue
        # If naip tile is not found output file name of missing tile and skip.
        if not os.path.exists(paths[i]):
            print('Missing file: ', paths[i])
            continue
        if os.path.exists(paths[i]):
            i = getattr(naip, index)(paths[i], outputs[i])


def batch_extra_trees(phy_id, smoothing=True):
    """
    This function enables batch classification of NAIP imagery using a
    sklearn Extra Trees supervised classification algorithm.
    ---
    Args:
        phy_id: int ::  Physio Id for the region to be processed.

    """

    workspace = config.workspace
    shp = config.naipqq_shp
    results_dir = config.results
    training_raster = config.training_raster
    fit_raster = config.training_fit_raster

    # Query region name, create input and output folder paths
    region = get_phyregs_name(phy_id)
    print(region)
    region_dir = '%s/%s' % (results_dir, region)
    in_dir = '%s/Inputs' % region_dir
    out_dir = '%s/Outputs' % region_dir
    if not os.path.exists(in_dir):
        raise IOError('Input directory does not exist.')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # Read training raster and corresponding raster file and shape to be trained
    y_raster = gdal.Open(training_raster)
    t = y_raster.GetRasterBand(1).ReadAsArray().astype(np.float32)
    x_raster = gdal.Open(fit_raster)
    n = x_raster.GetRasterBand(1).ReadAsArray().astype(np.float32)
    y = t[t > 0]
    X = n[t > 0]
    X = X.reshape(-1, 1)
    # Train Extra Trees Classifier
    clf = ExtraTreesClassifier(n_estimators=41, n_jobs=-1,
                               max_features=None,
                               min_samples_leaf=5, class_weight={1: 2, 2: 0.5})
    ras = clf.fit(X, y)

    # Open naip_qq shapefile and iterate over attributes to select naip tiles
    # in desired phy_id.
    src = ogr.Open(shp)
    lyr = src.GetLayer()
    FileName = []
    phyregs = []
    filtered = []
    paths = []
    query = '%d' % phy_id
    outputs = []
    for i in lyr:
        FileName.append(i.GetField('FileName'))
        phyregs.append(str(i.GetField('PHYSIO_ID')))
    # Get raw file names from naip_qq layer by iterating over phyregs list and
    # retreving corresponding file name from filenames list.
    for j in range(len(phyregs)):
        if query == phyregs[j]:
            filtered.append(FileName[j])
    for i in range(len(filtered)):
        # Edit filenames to get true file names, and create output filenames and
        # paths.
        file = '%s%s' % ('arvi_', filtered[i])
        filename = '%s.tif' % file[:-13]
        in_path = '%s/%s' % (in_dir, filename)
        out_file = '%s/%s%s' % (out_dir, 'c_', filename)
        outputs.append(out_file)
        paths.append(in_path)
        if os.path.exists(out_file):
            continue
        # Check if input file exists
        if not os.path.exists(paths[i]):
            print('Missing file: ', paths[i])
            continue
        if os.path.exists(paths[i]):
            # If input file exists open with gdal and convert to NumPy array.
            r = gdal.Open(paths[i])
            class_raster = r.GetRasterBand(1).ReadAsArray().astype(
                np.float32)
            class_array = class_raster.reshape(-1, 1)
            # Apply classification
            ras_pre = ras.predict(class_array)
            # Convert back to original shape and make data type Byte
            ras_final = ras_pre.reshape(class_raster.shape)
            ras_byte = ras_final.astype(dtype=np.byte)
            if smoothing:
                # If smoothing = True, apply SciPy median_filter to array and
                # then save.
                smooth_ras = ndimage.median_filter(ras_byte, size=5)
                driver = gdal.GetDriverByName('GTiff')
                metadata = driver.GetMetadata()
                shape = class_raster.shape
                dst_ds = driver.Create(outputs[i],
                                       shape[1],
                                       shape[0],
                                       1,
                                       gdal.GDT_Byte, ['NBITS=2'])
                proj = r.GetProjection()
                geo = r.GetGeoTransform()
                dst_ds.SetGeoTransform(geo)
                dst_ds.SetProjection(proj)
                dst_ds.GetRasterBand(1).WriteArray(smooth_ras)
                dst_ds.FlushCache()
                dst_ds = None
            if not smoothing:
                # If smoothing = False, save numpy array as raster with out
                # smoothing
                driver = gdal.GetDriverByName('GTiff')
                metadata = driver.GetMetadata()
                shape = class_raster.shape
                dst_ds = driver.Create(outputs[i],
                                       shape[1],
                                       shape[0],
                                       1,
                                       gdal.GDT_Byte, ['NBITS=2'])
                proj = r.GetProjection()
                geo = r.GetGeoTransform()
                dst_ds.SetGeoTransform(geo)
                dst_ds.SetProjection(proj)
                dst_ds.GetRasterBand(1).WriteArray(ras_byte)
                dst_ds.FlushCache()
                dst_ds = None


def clip_reproject_classified_tiles(phy_id):
    """
    This fucntion clips and reprojects all classified to their respective
    seamlines and the desired projection

    Args:
        phy_id: int ::  Physio Id for the region to be processed.
    """

    workspace = config.workspace
    shp = config.naipqq_shp
    clip_shp = config.clip_naip
    results_dir = config.results
    proj = config.proj

    region = get_phyregs_name(phy_id)
    print(region)
    region_dir = '%s/%s' % (results_dir, region)
    in_dir = '%s/Outputs' % region_dir
    out_dir = '%s/Outputs' % region_dir
    if not os.path.exists(in_dir):
        raise IOError('Input directory does not exist.')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    gdal.PushErrorHandler('CPLQuietErrorHandler')
    gdal.UseExceptions()
    gdal.AllRegister()
    np.seterr(divide='ignore', invalid='ignore')

    src = ogr.Open(shp)
    lyr = src.GetLayer()
    FileName = []
    phyregs = []
    filtered = []
    paths = []
    query = '%d' % phy_id
    outputs = []
    for i in lyr:
        FileName.append(i.GetField('FileName'))
        phyregs.append(str(i.GetField('PHYSIO_ID')))
    # Get raw file names from naip_qq layer by iterating over phyregs list and
    # retreving corresponding file name from filenames list.
    for j in range(len(phyregs)):
        if query == phyregs[j]:
            filtered.append(FileName[j])
    for i in range(len(filtered)):
        # Edit filenames to get true file names, and create output filenames and
        # paths.
        file = '%s%s' % ('c_arvi_', filtered[i])
        filename = '%s.tif' % file[:-13]
        in_path = '%s/%s' % (in_dir, filename)
        out_file = '%s/%s%s' % (out_dir, 'cl_', filename)
        where = "FileName = '%s'" % filtered[i]
        result = gdal.Warp(out_file, in_path, dstNodata=3, dstSRS=proj,
                           xRes=1, yRes=1, cutlineDSName=clip_shp,
                           cutlineWhere=where,
                           cropToCutline=True, outputType=gdal.GDT_Byte,
                           creationOptions=["NBITS=2"]
                           )
        result = None


def mosaic_tiles(phy_id):
    """
    This function mosaics all classified NAIP tiles within a physiographic
    region using gdal_merge.py
    ---
    Args:
        phy_id: int ::  Physio Id for the region to be processed.

    """
    shp = config.naipqq_shp
    naip_dir = config.naip_dir
    results_dir = config.results
    proj = config.proj

    region = get_phyregs_name(phy_id)
    print(region)
    region_dir = '%s/%s' % (results_dir, region)
    dir_path = '%s/Outputs' % (region_dir)
    src = ogr.Open(shp)
    lyr = src.GetLayer()
    FileName = []
    phyregs = []
    filtered = []
    query = '%d' % phy_id
    inputs = []
    for i in lyr:
        FileName.append(i.GetField('FileName'))
        phyregs.append(str(i.GetField('PHYSIO_ID')))
    # Get raw file names from naip_qq layer by iterating over phyregs list and
    # retreving corresponding file name from filenames list.
    for j in range(len(phyregs)):
        if query == phyregs[j]:
            filtered.append(FileName[j])
    for i in range(len(filtered)):
        # Edit filenames to get true file names, and create output filenames and
        # paths.
        file = filtered[i]
        filename = '%s.tif' % file[:-13]
        in_file = '%s/%s%s' % (dir_path, 'cl_c_arvi_', filename)
        out_file = '%s/%s%s.tif' % (dir_path, 'mosaic_', region)
        inputs.append(in_file)
        # Check if input file exists
        if not os.path.exists(inputs[i]):
            print('Missing file: ', inputs[i])
            continue

    inputs_string = " ".join(inputs)
    gdal_merge = "gdal_merge.py -co NBITS=2 -n 3 -init 3 -o %s -of gtiff %s" % (
        out_file, inputs_string)
    os.system(gdal_merge)


def clip_mosaic(phy_id):
    shp = config.phyreg_lyr
    results_dir = config.results

    region = get_phyregs_name(phy_id)
    print(region)
    region_dir = '%s/%s' % (results_dir, region)
    dir_path = '%s/Outputs' % (region_dir)
    input_raster_name = 'mosaic_%s.tif' % region
    in_raster = '%s/%s' % (dir_path, input_raster_name)
    out_raster = '%s/clipped_%s' % (dir_path, input_raster_name)

    where = "PHYSIO_ID = %d" % phy_id

    warp = gdal.Warp(out_raster, in_raster, xRes=1, yRes=1, cutlineDSName=shp,
                     cutlineWhere=where, cropToCutline=True,
                     srcNodata='3', dstNodata='3',
                     outputType=gdal.GDT_Byte, creationOptions=["NBITS=2"],
                     dstSRS=proj)


def create_canopy_dataset(phy_id):
    """
    This function is a wrapper function run every step to make a canopy dataset.
    Args:
        phy_id: int ::  Physio Id for the region to be processed.
    """
    ARVI(phy_id)
    batch_extra_trees(phy_id)
    clip_reproject_classified_tiles(phy_id)
    mosaic_tiles(phy_id)
    clip_mosaic(phy_id)
    print('Finished')
