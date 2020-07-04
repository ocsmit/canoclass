import os
from osgeo import ogr
from canoclass.batchnaip import config


def batch_mosaic(pid):
    """
    This function mosaics all classified NAIP tiles within a physiographic
    region using gdal_merge.py
    ---
    Args:
        phy_id: int ::  Physio Id for the region to be processed.

    """
    shp = config.naipqq_shp
    results_dir = config.results
    id_field = config.procid_field

    region_dir = '%s/%s' % (results_dir, pid)
    dir_path = '%s/Outputs' % (region_dir)
    src = ogr.Open(shp)
    lyr = src.GetLayer()
    FileName = []
    phyregs = []
    filtered = []
    query = '%d' % pid
    inputs = []
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
        file = filtered[i]
        filename = '%s.tif' % file[:-13]
        in_file = '%s/%s%s' % (dir_path, 'cl_c_arvi_', filename)
        out_file = '%s/%s%s.tif' % (dir_path, 'mosaic_', str(pid))
        inputs.append(in_file)
        # Check if input file exists
        if not os.path.exists(inputs[i]):
            print('Missing file: ', inputs[i])
            continue

    inputs_string = " ".join(inputs)
    print(inputs_string)
    gdal_merge = "gdal_merge.py -co NBITS=2 -n 3 -init 3 -o %s -of gtiff %s" % (
        out_file, inputs_string)
    os.system(gdal_merge)

