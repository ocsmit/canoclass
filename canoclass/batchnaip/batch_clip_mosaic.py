from osgeo import gdal
from canoclass.batchnaip import config

def batch_clip_mosaic(pid):
    shp = config.proc_region
    results_dir = config.results
    proj = config.proj
    id_field = config.procid_field

    region_dir = '%s/%s' % (results_dir, pid)
    dir_path = '%s/Outputs' % (region_dir)
    input_raster_name = 'mosaic_%s.tif' % pid
    in_raster = '%s/%s' % (dir_path, input_raster_name)
    out_raster = '%s/clipped_%s' % (dir_path, input_raster_name)

    where = "%s = %d" % (id_field, pid)

    warp = gdal.Warp(out_raster, in_raster, xRes=1, yRes=1, cutlineDSName=shp,
                     cutlineWhere=where, cropToCutline=True,
                     srcNodata='3', dstNodata='3',
                     outputType=gdal.GDT_Byte, creationOptions=["NBITS=2"],
                     dstSRS=proj)
