from osgeo import gdal
import numpy as np

def load_data(training_raster, training_fit_raster):

    y_raster = gdal.Open(training_raster)
    t = y_raster.GetRasterBand(1).ReadAsArray().astype(np.float64)
    x_raster = gdal.Open(training_fit_raster)
    n = x_raster.GetRasterBand(1).ReadAsArray().astype(np.float64)
    n[np.isnan(n)] = 0
    n_mask = np.ma.MaskedArray(n, mask=(n == 0))
    n_mask.reshape(n.shape)
    y = t[t > 0]
    X = n_mask[t > 0]
    X = X.reshape(-1, 1)

    return X, y