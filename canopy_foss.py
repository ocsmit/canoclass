# ==============================================================================
# Preprocessing Functions:
# ------------------------
#       Index Calculations:
#       -- ARVI(naip_dir, out_dir)
#       -- VARI(naip_dir, out_dir)
#       -- GRVI(naip_dir, out_dir)
#
#       Training Data Prep:
#       -- prepare_training_data(vector, ref_raster, out_raster, field='id')
# ==============================================================================

import os
from osgeo import gdal, ogr
import numpy as np
import config
from scipy import ndimage
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split



def get_naip_path(shp, phy_id, naip_dir):
    src = ogr.Open(shp)
    lyr = src.GetLayer()
    FileName = []
    phyregs = []
    filtered = []
    paths = []
    query = ',%d,' % phy_id
    for i in lyr:
        FileName.append(i.GetField('FileName'))
        phyregs.append(i.GetField('phyregs'))
    for j in range(len(phyregs)):
        if query in phyregs[j]:
            filtered.append(FileName[j])
    for i in range(len(filtered)):
        file = filtered[i]
        filename = '%s.tif' % file[:-13]
        folder = file[2:7]
        path = '%s/%s/%s' % (naip_dir, folder, filename)
        paths.append(path)
    return paths


def get_arvi_path(shp, phy_id, arvi_dir):
    src = ogr.Open(shp)
    lyr = src.GetLayer()
    FileName = []
    phyregs = []
    filtered = []
    paths = []
    query = ',%d,' % phy_id
    for i in lyr:
        FileName.append(i.GetField('FileName'))
        phyregs.append(i.GetField('phyregs'))
    for j in range(len(phyregs)):
        if query in phyregs[j]:
            filtered.append(FileName[j])
    for i in range(len(filtered)):
        file = '%s%s' % ('arvi_', filtered[i])
        filename = '%s.tif' % file[:-13]
        folder = file[2:7]
        path = '%s/%s' % (arvi_dir, filename)
        paths.append(path)
    return paths


def ARVI(phy_id):
    """
    This function walks through the input NAIP directory and performs the
    ARVI calculation on each naip geotiff file and saves each new ARVI
    geotiff in the output directory with the prefix 'arvi_'
    ---
    Args:
        naip_dir: Folder which contains all subfolders of naip imagery
        out_dir:  Folder in which all calculated geotiff's are saved
    """
    workspace = config.workspace
    shp = config.naipqq_shp
    naip_dir = config.naip_dir
    out_dir = config.arvi_dir

    naip_path = get_naip_path(shp, phy_id, naip_dir)

    if not os.path.exists(naip_dir):
        print('NAIP directory not found')
    if not os.path.exists(workspace):
        os.mkdir(workspace)
        os.mkdir(out_dir)

    # Create list with file names
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
    query = ',%d,' % phy_id
    outputs = []
    for i in lyr:
        FileName.append(i.GetField('FileName'))
        phyregs.append(i.GetField('phyregs'))
    for j in range(len(phyregs)):
        if query in phyregs[j]:
            filtered.append(FileName[j])
    for i in range(len(filtered)):
        file = filtered[i]
        filename = '%s.tif' % file[:-13]
        arvi_file = 'arvi_%s' % filename
        folder = file[2:7]
        in_path = '%s/%s/%s' % (naip_dir, folder, filename)
        out_path = '%s/%s' % (out_dir, arvi_file)
        outputs.append(out_path)
        paths.append(in_path)
        if os.path.exists(outputs[i]):
            continue
        if not os.path.exists(outputs[i]):
            # Open with gdal & create numpy arrays
            naip = gdal.Open(paths[i])
            red_band = naip.GetRasterBand(1).ReadAsArray() \
                .astype(np.float32)
            blue_band = naip.GetRasterBand(3).ReadAsArray() \
                .astype(np.float32)
            nir_band = naip.GetRasterBand(4).ReadAsArray() \
                .astype(np.float32)
            snap = naip
            # Perform Calculation
            a = (nir_band - (2 * red_band) + blue_band)
            b = (nir_band + (2 * red_band) + blue_band)
            arvi = a / b
            # Save Raster
            driver = gdal.GetDriverByName('GTiff')
            metadata = driver.GetMetadata()
            shape = arvi.shape
            dst_ds = driver.Create(outputs[i],
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
            print(outputs[i])


def nVARI(naip_dir, out_dir):
    """
    This function walks through the input NAIP directory and performs the
    VARI calculation on each naip geotiff file and saves each new VARI
    geotiff in the output directory with the prefix 'arvi_'
    ---
    Args:
        naip_dir: Folder which contains all subfolders of naip imagery
        out_dir: Folder in which all calculated geotiff's are saved
    """
    if not os.path.exists(naip_dir):
        print('NAIP directory not found')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    gdal.PushErrorHandler('CPLQuietErrorHandler')
    gdal.UseExceptions()
    gdal.AllRegister()
    np.seterr(divide='ignore', invalid='ignore')

    def norm(array):
        array_min, array_max = array.min(), array.max()
        return ((1 - 0) * ((array - array_min) / (array_max - array_min))) + 1

    for dir, subdir, files in os.walk(naip_dir):
        for f in files:
            name = 'vari_%s' % f
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


def prepare_training_data(vector, ref_raster, out_raster, field='id'):
    """
    This function converts the training data shapefile into a raster to allow
    the training data to be applied for classification
    ---
    Args:
        vector:
        ref_raster:
        out_raster:
        field:
    """
    # TODO: Allow for training data to have 0 and 1 as values

    snap = gdal.Open(ref_raster)
    shp = ogr.Open(vector)
    layer = shp.GetLayer()

    xy = snap.GetRasterBand(1).ReadAsArray().astype(np.float32).shape

    driver = gdal.GetDriverByName('GTiff')
    metadata = driver.GetMetadata()
    dst_ds = driver.Create(out_raster,
                           xsize=xy[1],
                           ysize=xy[0],
                           bands=1,
                           eType=gdal.GDT_Byte)
    proj = snap.GetProjection()
    geo = snap.GetGeoTransform()
    dst_ds.SetGeoTransform(geo)
    dst_ds.SetProjection(proj)
    if field is None:
        gdal.RasterizeLayer(dst_ds, [1], layer, None)
    else:
        OPTIONS = ['ATTRIBUTE=' + field]
        gdal.RasterizeLayer(dst_ds, [1], layer, None, options=OPTIONS)
    dst_ds.FlushCache()
    dst_ds = None

    print('Vector to raster complete.')
    return out_raster

# ==============================================================================
# Classification Functions:
# -------------------------
#       -- split_data(training_raster, training_fit_raster)
#       -- tune_hyperparameter(training_raster, training_fit_raster)
#       Random Forests:
#       -- random_forests_class(training_raster, training_fit_raster, in_raster,
#                               out_tiff, smoothing=True)
#       -- batch_random_forests(in_directory, training_raster, fit_raster,
#                               out_directory, smoothing=True)
#       Extra Trees:
#       -- extra_trees_class(training_raster, training_fit_raster, in_raster,
#                            out_tiff, smoothing=True):
#       -- batch_extra_trees(in_directory, training_raster, fit_raster,
#                            out_directory, smoothing=True):
# ==============================================================================


def split_data(training_raster, training_fit_raster):

    y_raster = gdal.Open(training_raster)
    t = y_raster.GetRasterBand(1).ReadAsArray().astype(np.float32)
    x_raster = gdal.Open(training_fit_raster)
    n = x_raster.GetRasterBand(1).ReadAsArray().astype(np.float32)
    y = t[t > 0]
    X = n[t > 0]
    X = X.reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =
    0.33)

    return X_train, X_test, y_train, y_test


def tune_hyperparameter(training_raster, training_fit_raster):

    y_raster = gdal.Open(training_raster)
    t = y_raster.GetRasterBand(1).ReadAsArray().astype(np.float32)
    x_raster = gdal.Open(training_fit_raster)
    n = x_raster.GetRasterBand(1).ReadAsArray().astype(np.float32)
    y = t[t > 0]
    X = n[t > 0]
    X = X.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =
    0.33)

    n_estimators = [int(x) for x in np.linspace(start=10, stop=150, num=10)]
    min_samples_leaf = [int(x) for x in np.linspace(start=5, stop=500, num=10)]
    random_grid = {
        'n_estimators': n_estimators,
        'min_samples_leaf': min_samples_leaf
    }
    weight = [{1: 1, 2: 2}]
    etc = ExtraTreesClassifier(n_estimators=100, n_jobs=-1,
                               max_features=None,
                               min_samples_leaf=10, class_weight={1: 2, 2: 0.5})
    clf = RandomizedSearchCV(etc, random_grid, random_state=0, verbose=3)
    clf.fit(X_train, y_train)

    print(clf.best_params_)


def random_forests_class(training_raster, training_fit_raster, in_raster,
                         out_tiff, smoothing=True):
    """
    This function enables classification of NAIP imagery using a sklearn Random
    Forests supervised classification algorithm.
    ---
    Args:
        training_fit_raster:
        training_raster: Rasterized training data
        in_raster: Raster training raster will be applied to
        out_tiff: Final output classified raster
        smoothing: True :: applies median filter to output classified raster
    """

    y_raster = gdal.Open(training_raster)
    t = y_raster.GetRasterBand(1).ReadAsArray().astype(np.float32)
    x_raster = gdal.Open(training_fit_raster)
    n = x_raster.GetRasterBand(1).ReadAsArray().astype(np.float32)
    y = t[t > 0]
    X = n[t > 0]
    X = X.reshape(-1, 1)
    clf = RandomForestClassifier(n_estimators=50, n_jobs=2,
                                 max_features='sqrt',
                                 min_samples_leaf=10)
    ras = clf.fit(X, y)
    r = gdal.Open(in_raster)
    class_raster = r.GetRasterBand(1).ReadAsArray().astype(np.float32)
    class_array = class_raster.reshape(-1, 1)
    ras_pre = ras.predict(class_array)
    ras_final = ras_pre.reshape(class_raster.shape)
    ras_byte = ras_final.astype(dtype=np.byte)

    if smoothing:
        smooth_ras = ndimage.median_filter(ras_byte, size=5)
        driver = gdal.GetDriverByName('GTiff')
        metadata = driver.GetMetadata()
        shape = class_raster.shape
        dst_ds = driver.Create(out_tiff,
                               xsize=shape[1],
                               ysize=shape[0],
                               bands=1,
                               eType=gdal.GDT_Byte)
        proj = r.GetProjection()
        geo = r.GetGeoTransform()
        dst_ds.SetGeoTransform(geo)
        dst_ds.SetProjection(proj)
        dst_ds.GetRasterBand(1).WriteArray(smooth_ras)
        dst_ds.FlushCache()
        dst_ds = None
    if not smoothing:
        driver = gdal.GetDriverByName('GTiff')
        metadata = driver.GetMetadata()
        shape = class_raster.shape
        dst_ds = driver.Create(out_tiff,
                               xsize=shape[1],
                               ysize=shape[0],
                               bands=1,
                               eType=gdal.GDT_Byte)
        proj = r.GetProjection()
        geo = r.GetGeoTransform()
        dst_ds.SetGeoTransform(geo)
        dst_ds.SetProjection(proj)
        dst_ds.GetRasterBand(1).WriteArray(ras_byte)
        dst_ds.FlushCache()
        dst_ds = None
    print(out_tiff)


def batch_random_forests(in_directory, training_raster, fit_raster,
                         out_directory, smoothing=True):
    """
    This function enables batch classification of NAIP imagery using a
    sklearn Ec supervised classification algorithm.
    ---
    Args:
        in_directory: Input naip directory
        training_raster: Rasterized training data
        fit_raster: Raster training raster will be applied to
        out_directory: output directory for classified imagery
        smoothing: True :: applies median filter to output classified raster
    """
    y_raster = gdal.Open(training_raster)
    t = y_raster.GetRasterBand(1).ReadAsArray().astype(np.float32)
    x_raster = gdal.Open(fit_raster)
    n = x_raster.GetRasterBand(1).ReadAsArray().astype(np.float32)
    y = t[t > 0]
    X = n[t > 0]
    X = X.reshape(-1, 1)

    clf = RandomForestClassifier(n_estimators=50, n_jobs=2,
                                 max_features='sqrt',
                                 min_samples_leaf=10)
    ras = clf.fit(X, y)
    if not os.path.exists(out_directory):
        os.mkdir(out_directory)
    for dir, subdir, files in os.walk(in_directory):
        for f in files:
            input_raster = os.path.join(in_directory, f)
            output = os.path.join(out_directory, 'erf_' + f)
            r = gdal.Open(input_raster)
            class_raster = r.GetRasterBand(1).ReadAsArray().astype(np.float32)
            class_array = class_raster.reshape(-1, 1)
            ras_pre = ras.predict(class_array)
            ras_final = ras_pre.reshape(class_raster.shape)
            ras_byte = ras_final.astype(dtype=np.byte)
            if os.path.exists(output):
                continue
            if not os.path.exists(output):
                if smoothing:
                    smooth_ras = ndimage.median_filter(ras_byte, size=5)
                    driver = gdal.GetDriverByName('GTiff')
                    metadata = driver.GetMetadata()
                    shape = class_raster.shape
                    dst_ds = driver.Create(output,
                                           xsize=shape[1],
                                           ysize=shape[0],
                                           bands=1,
                                           eType=gdal.GDT_Byte)
                    proj = r.GetProjection()
                    geo = r.GetGeoTransform()
                    dst_ds.SetGeoTransform(geo)
                    dst_ds.SetProjection(proj)
                    dst_ds.GetRasterBand(1).WriteArray(smooth_ras)
                    dst_ds.FlushCache()
                    dst_ds = None
                if not smoothing:
                    driver = gdal.GetDriverByName('GTiff')
                    metadata = driver.GetMetadata()
                    shape = class_raster.shape
                    dst_ds = driver.Create(output,
                                           xsize=shape[1],
                                           ysize=shape[0],
                                           bands=1,
                                           eType=gdal.GDT_Byte)
                    proj = r.GetProjection()
                    geo = r.GetGeoTransform()
                    dst_ds.SetGeoTransform(geo)
                    dst_ds.SetProjection(proj)
                    dst_ds.GetRasterBand(1).WriteArray(ras_byte)
                    dst_ds.FlushCache()
                    dst_ds = None
                print(output)
    print('Complete.')


def extra_trees_class(training_raster, training_fit_raster, in_raster,
                      out_tiff, smoothing=True):
    """
    This function enables classification of NAIP imagery using a sklearn Random
    Forests supervised classification algorithm.
    ---
    Args:
        training_fit_raster:
        training_raster: Rasterized training data
        in_raster: Raster training raster will be applied to
        out_tiff: Final output classified raster
        smoothing: True :: applies median filter to output classified raster
    """
    y_raster = gdal.Open(training_raster)
    t = y_raster.GetRasterBand(1).ReadAsArray().astype(np.float32)
    x_raster = gdal.Open(training_fit_raster)
    n = x_raster.GetRasterBand(1).ReadAsArray().astype(np.float32)
    y = t[t > 0]
    X = n[t > 0]
    X = X.reshape(-1, 1)
    clf = ExtraTreesClassifier(n_estimators=100, n_jobs=-1,
                               max_features=None,
                               min_samples_leaf=10, class_weight={1: 2, 2: 0.5})
    ras = clf.fit(X, y)
    r = gdal.Open(in_raster)
    class_raster = r.GetRasterBand(1).ReadAsArray().astype(np.float32)
    class_array = class_raster.reshape(-1, 1)
    ras_pre = ras.predict(class_array)
    ras_final = ras_pre.reshape(class_raster.shape)
    ras_byte = ras_final.astype(dtype=np.byte)
    if smoothing:
        smooth_ras = ndimage.median_filter(ras_byte, size=5)
        driver = gdal.GetDriverByName('GTiff')
        metadata = driver.GetMetadata()
        shape = class_raster.shape
        dst_ds = driver.Create(out_tiff,
                               xsize=shape[1],
                               ysize=shape[0],
                               bands=1,
                               eType=gdal.GDT_Byte)
        proj = r.GetProjection()
        geo = r.GetGeoTransform()
        dst_ds.SetGeoTransform(geo)
        dst_ds.SetProjection(proj)
        dst_ds.GetRasterBand(1).WriteArray(smooth_ras)
        dst_ds.FlushCache()
        dst_ds = None
    if not smoothing:
        driver = gdal.GetDriverByName('GTiff')
        metadata = driver.GetMetadata()
        shape = class_raster.shape
        dst_ds = driver.Create(out_tiff,
                               xsize=shape[1],
                               ysize=shape[0],
                               bands=1,
                               eType=gdal.GDT_Byte)
        proj = r.GetProjection()
        geo = r.GetGeoTransform()
        dst_ds.SetGeoTransform(geo)
        dst_ds.SetProjection(proj)
        dst_ds.GetRasterBand(1).WriteArray(ras_byte)
        dst_ds.FlushCache()
        dst_ds = None

    print(out_tiff)


def batch_extra_trees(in_directory, training_raster, fit_raster, out_directory,
                      smoothing=True):
    """
    This function enables batch classification of NAIP imagery using a
    sklearn Extra Trees supervised classification algorithm.
    ---
    Args:
        in_directory: Input naip directory
        training_raster: Rasterized training data
        fit_raster: Raster training raster will be applied to
        out_directory: output directory for classified imagery
        smoothing: True :: applies median filter to output classified raster
    """
    y_raster = gdal.Open(training_raster)
    t = y_raster.GetRasterBand(1).ReadAsArray().astype(np.float32)
    x_raster = gdal.Open(fit_raster)
    n = x_raster.GetRasterBand(1).ReadAsArray().astype(np.float32)
    y = t[t > 0]
    X = n[t > 0]
    X = X.reshape(-1, 1)
    clf = ExtraTreesClassifier(n_estimators=100, n_jobs=-1,
                               max_features=None,
                               min_samples_leaf=10, class_weight={1: 2, 2: 0.5})
    ras = clf.fit(X, y)
    if not os.path.exists(out_directory):
        os.mkdir(out_directory)
    for dir, subdir, files in os.walk(in_directory):
        for f in files:
            input_raster = os.path.join(in_directory, f)
            output = os.path.join(out_directory, 'rf_' + f)
            if os.path.exists(output):
                continue
            if not os.path.exists(output):
                r = gdal.Open(input_raster)
                class_raster = r.GetRasterBand(1).ReadAsArray().astype(
                    np.float32)
                class_array = class_raster.reshape(-1, 1)
                ras_pre = ras.predict(class_array)
                ras_final = ras_pre.reshape(class_raster.shape)
                ras_byte = ras_final.astype(dtype=np.byte)
                if smoothing:
                    smooth_ras = ndimage.median_filter(ras_byte, size=5)
                    driver = gdal.GetDriverByName('GTiff')
                    metadata = driver.GetMetadata()
                    shape = class_raster.shape
                    dst_ds = driver.Create(output,
                                           xsize=shape[1],
                                           ysize=shape[0],
                                           bands=1,
                                           eType=gdal.GDT_Byte)
                    proj = r.GetProjection()
                    geo = r.GetGeoTransform()
                    dst_ds.SetGeoTransform(geo)
                    dst_ds.SetProjection(proj)
                    dst_ds.GetRasterBand(1).WriteArray(smooth_ras)
                    dst_ds.FlushCache()
                    dst_ds = None
                if not smoothing:
                    driver = gdal.GetDriverByName('GTiff')
                    metadata = driver.GetMetadata()
                    shape = class_raster.shape
                    dst_ds = driver.Create(output,
                                           xsize=shape[1],
                                           ysize=shape[0],
                                           bands=1,
                                           eType=gdal.GDT_Byte)
                    proj = r.GetProjection()
                    geo = r.GetGeoTransform()
                    dst_ds.SetGeoTransform(geo)
                    dst_ds.SetProjection(proj)
                    dst_ds.GetRasterBand(1).WriteArray(ras_byte)
                    dst_ds.FlushCache()
                    dst_ds = None
                print(output)
    print('Complete.')
