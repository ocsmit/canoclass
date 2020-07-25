from osgeo import gdal
import numpy as np

def load_data(training_raster, training_fit_raster):
    """
    Returns properly shaped X, y training data for classification

    Parameters
    ----------
        training_raster : str, filename
            The rasterized training data.
        training_fit_raster : str, filename
            The vegetation index raster that the rasterized
            training data will be fit with.

    Returns
    -------
        X, y: array
            Data array to load into classifier

    """

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
