import os
from osgeo import gdal, ogr
import numpy as np
from canoclass.batchnaip import config
from scipy import ndimage
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from rindcalc import naip
from canoclass.utils import load_data


def batchIndex(pid, index='ARVI'):
    """
    This function walks through the input NAIP directory and performs the
    ARVI calculation on each naip geotiff file and saves each new ARVI
    geotiff in the output directory with the prefix 'arvi_'
    ---
    Args:
        phy_id: int ::  Physio Id for the region to be processed.
    """
    shp = config.naipqq_shp
    naip_dir = config.naip_dir
    results_dir = config.results
    id_field = config.procid_field

    if not os.path.exists(naip_dir):
        print('NAIP directory not found')
    region_dir = '%s/%s' % (results_dir, str(pid))
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
    query = '%d' % pid
    outputs = []
    # Query is done by iterating over list of entire naip_qq shapefile.
    # ogr.SetAttributeFilter throws SQL expression error due to needed commas
    # around phy_id.
    for i in lyr:
        FileName.append(i.GetField('FileName'))
        phyregs.append(str(i.GetField(id_field)))
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
