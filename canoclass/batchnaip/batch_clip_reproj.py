import os
from osgeo import gdal, ogr
import numpy as np
from canoclass.batchnaip import config


def batch_clip_reproject(pid):
    """
    This fucntion clips and reprojects all classified to their respective
    seamlines and the desired projection

    Args:
        phy_id: int ::  Physio Id for the region to be processed.
    """

    shp = config.naipqq_shp
    clip_shp = config.clip_naip
    results_dir = config.results
    proj = config.proj
    id_field = config.procid_field

    region_dir = '%s/%s' % (results_dir, pid)
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
    query = '%d' % pid
    for i in lyr:
        FileName.append(i.GetField('FileName'))
        phyregs.append(str(i.GetField(id_field)))
    # Get raw file names from naip_qq layer by iterating over phyregs list and
    # retreving corresponding file name from filenames list.
    for j in range(len(phyregs)):
        if query == phyregs[j]:
            filtered.append(FileName[j])
    for i in range(len(filtered)):
        # Edit filenames to get true file names
        # , and create output filenames and
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
