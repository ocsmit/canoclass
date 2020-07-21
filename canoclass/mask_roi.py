from osgeo import gdal


def mask_roi(mask_shp, in_tif, out_tif):
    """
    This function allows the input rasters to be clipped to the extent of the
    masking shape file. The default resolution and projection is preserved.
    ---
    Args:
        mask_shp: Path to shapefile with which to mask input raster
        in_tif: Path to raster to mask
        out_tif: File to save the output masked raster to.
    """

    mask = gdal.Warp(out_tif, in_tif, cutlineDSName=mask_shp, cropToCutline=True)

    del(mask)

