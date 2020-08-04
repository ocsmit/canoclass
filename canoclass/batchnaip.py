# ==============================================================================
# Title: batchnaip.py
# Author: Owen Smith, University of North Georgia
# Canopy data creation config:
# -----------------------------
# All proccess functions rely on the configurations set within this
# file to run.
#
# ==============================================================================
import os
from osgeo import gdal, ogr
import numpy as np
from rindcalc import naip
from scipy import ndimage
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from canoclass.load_data import load_data


class Batch:

    def __init__(self, workspace, naip_dir,
                 roi_shp, roi_id,
                 naipqq_clip, naipqq_query, training_raster,
                 training_fit_raster, projection):

        """
        Intialize the configuration for batch NAIP processing.

        Parameters
        ----------

            workspace : str, path
                Path wherein all data is contained will be read from
                and output into. The Data folder containing all input and
                reference data is contained within the workspace folder.
            naip_dir : str, path
                Path to directory where NAIP imagery is contained.
            roi_shp : str, filename
                The region of interest shapefile.
            roi_id : str
                The field from which to query the ROI.
            naipqq_clip : str, filename
                The original NAIP QQ shapfile that will allow NAIP
                tiles to be clipped to their QQ extent.
            naipqq_query : str, filename
                The NAIP QQ shapefile joined with the ROI shapefile.
                The roi_id will be queried against this shapefile to know which
                NAIP tiles to process.
            training_raster : str, filename
                The rasterized training data.
            training_fit_raster : str, filename
                The vegetation index raster that the rasterized
                training data will be fit with.
            projection : str
                The final projection the data will be in. Must follow
                GDAL formating. eg: "EPSG:5070"

        Attributes
        ----------
            config : dict
                Dictionary of all config parameters

        """

        data_dir = "%s/Data" % workspace
        results_dir = "%s/Results" % workspace

        config = {"naip_dir": naip_dir,
                  "data_dir": data_dir,
                  "results_dir": results_dir,
                  "class_dir": '%s/%%s/Outputs' % results_dir,
                  "roi_shp": "%s/%s" % (data_dir, roi_shp),
                  "roi_id": roi_id,
                  "naipqq_clip": "%s/%s" % (data_dir, naipqq_clip),
                  "naipqq_query": "%s/%s" % (data_dir, naipqq_query),
                  "training_raster": "%s/%s" % (data_dir, training_raster),
                  "training_fit_raster": "%s/%s" % (data_dir,
                                                    training_fit_raster),
                  "projection": projection}

        self.config = config

    def batch_index(self, pid, index='ARVI'):
        """
        This function walks through the input NAIP directory and performs the
        vegetation index calculation on each naip geotiff file and saves each
        new index geotiff in the output directory.

        Parameters
        ----------

            phy_id : int
                roi_id number for the region to be processed.
            index : str, default="ARVI"
                Which vegetation index to compute with rindcalc
        """

        config = self.config

        shp = config["naipqq_query"]
        naip_dir = config["naip_dir"]
        results_dir = config["results_dir"]
        id_field = config["roi_id"]

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

    def batch_rf_class(self, pid, smoothing=True, class_parameters=None):
        """
        This function enables batch classification of NAIP imagery using a
        sklearn Random Forests supervised classification algorithm.

        Parameters
        ----------
            phy_id : int
                roi_id number for the region to be processed.
            smoothing : bool, defualt=True
                Applies a 3x3 median filter to output classified raster.
            class_parameters : dict
                arguments for Scikit-learns ET Classifier
                {"n_estimators": 100, "criterion": 'gini', 
                "max_depth": None,
                 "min_samples_split": 2, "min_samples_leaf": 1,
                 "min_weight_fraction_leaf": 0.0, "max_features": 'auto',
                 "max_leaf_nodes": None, "min_impurity_decrease": 0.0,
                 "min_impurity_split": None, "bootstrap": True,
                 "oob_score": False, "n_jobs": None, "random_state": None,
                 "verbose": 0, "warm_start": False, "class_weight": None,
                 "ccp_alpha": 0.0, "max_samples": None}

        """

        config = self.config

        shp = config["naipqq_query"]
        results_dir = config["results_dir"]
        training_raster = config["training_raster"]
        training_fit_raster = config["training_fit_raster"]
        id_field = config["roi_id"]

        # Query region name, create input and output folder paths
        region_dir = '%s/%s' % (results_dir, pid)
        in_dir = '%s/Inputs' % region_dir
        out_dir = '%s/Outputs' % region_dir
        if not os.path.exists(in_dir):
            raise IOError('Input directory does not exist.')
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        # Read training & fit raster file and shape to be trained
        X, y = load_data(training_raster, training_fit_raster)

        # Train Random Forests Classifier
        if class_parameters is None:
            parameters = {"n_estimators": 100, "criterion": 'gini',
                          "max_depth": None,
                          "min_samples_split": 2, "min_samples_leaf": 1,
                          "min_weight_fraction_leaf": 0.0,
                          "max_features": 'auto',
                          "max_leaf_nodes": None, "min_impurity_decrease": 0.0,
                          "min_impurity_split": None, "bootstrap": True,
                          "oob_score": False, "n_jobs": None,
                          "random_state": None,
                          "verbose": 0, "warm_start": False,
                          "class_weight": None,
                          "ccp_alpha": 0.0, "max_samples": None}
            clf = RandomForestClassifier(**parameters)
        else:
            parameters = class_parameters
            clf = RandomForestClassifier(**parameters)

        ras = clf.fit(X, y)

        # Open naip_qq shapefile and iterate over attributes to select naip tiles
        # in desired pid.
        src = ogr.Open(shp)
        lyr = src.GetLayer()
        FileName = []
        phyregs = []
        filtered = []
        paths = []
        query = '%d' % pid
        outputs = []
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
            # create output filenames and
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
                class_raster[np.isnan(class_raster)] = 0
                class_mask = np.ma.MaskedArray(class_raster,
                                               mask=(class_raster == 0))
                class_mask.reshape(class_raster.shape)
                class_array = class_mask.reshape(-1, 1)

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

    def batch_et_class(self, pid, smoothing=True, class_parameters=None):
        """
        This function enables batch classification of NAIP imagery using a
        sklearn Extra Trees supervised classification algorithm.

        Parameters
        ----------
            phy_id : int
                roi_id number for the region to be processed.
            smoothing : bool, defualt=True
                Applies a 3x3 median filter to output classified raster.
            class_parameters : dict
                arguments for Scikit-learns ET Classifier
                
                {"n_estimators": 100, "criterion": 'gini',
                 "max_depth": None,
                 "min_samples_split": 2, "min_samples_leaf": 1,
                 "min_weight_fraction_leaf": 0.0, "max_features": 'auto',
                 "max_leaf_nodes": None, "min_impurity_decrease": 0.0,
                 "min_impurity_split": None, "bootstrap": False,
                 "oob_score": False, "n_jobs": None, "random_state": None,
                 "verbose": 0, "warm_start": False, "class_weight": None,
                 "ccp_alpha": 0.0, "max_samples": None}

        """

        config = self.config

        shp = config["naipqq_query"]
        results_dir = config["results_dir"]
        training_raster = config["training_raster"]
        training_fit_raster = config["training_fit_raster"]
        id_field = config["roi_id"]

        # Query region name, create input and output folder paths
        region_dir = '%s/%s' % (results_dir, pid)
        in_dir = '%s/Inputs' % region_dir
        out_dir = '%s/Outputs' % region_dir
        if not os.path.exists(in_dir):
            raise IOError('Input directory does not exist.')
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        # Read training & fit raster file and shape to be trained
        X, y = load_data(training_raster, training_fit_raster)

        # Train Extra Trees Classifier
        if class_parameters is None:
            parameters = {"n_estimators": 100, "criterion": 'gini',
                          "max_depth": None,
                          "min_samples_split": 2, "min_samples_leaf": 1,
                          "min_weight_fraction_leaf": 0.0,
                          "max_features": 'auto',
                          "max_leaf_nodes": None, "min_impurity_decrease": 0.0,
                          "min_impurity_split": None, "bootstrap": False,
                          "oob_score": False, "n_jobs": None,
                          "random_state": None,
                          "verbose": 0, "warm_start": False,
                          "class_weight": None,
                          "ccp_alpha": 0.0, "max_samples": None}
            clf = ExtraTreesClassifier(**parameters)
        else:
            parameters = class_parameters
            clf = ExtraTreesClassifier(**parameters)

        ras = clf.fit(X, y)

        # Open naip_qq shapefile and iterate over attributes to select naip tiles
        # in desired pid.
        src = ogr.Open(shp)
        lyr = src.GetLayer()
        FileName = []
        phyregs = []
        filtered = []
        paths = []
        query = '%d' % pid
        outputs = []
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
            # create output filenames and
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

                class_raster[np.isnan(class_raster)] = 0
                class_mask = np.ma.MaskedArray(class_raster,
                                               mask=(class_raster == 0))
                class_mask.reshape(class_raster.shape)
                class_array = class_mask.reshape(-1, 1)

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

    def batch_clip_reproject(self, pid):
        """
        This fucntion clips and reprojects all classified to their respective
        seamlines and the desired projection

        Parameters
        ----------
            phy_id : int
                roi_id number for the region to be processed.
        """

        config = self.config

        shp = config["naipqq_query"]
        clip_shp = config["naipqq_clip"]
        results_dir = config["results_dir"]
        proj = config["projection"]
        id_field = config["roi_id"]

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

    def batch_mosaic(self, pid):
        """
        This function mosaics all classified NAIP tiles within a physiographic
        region using gdal_merge.py


        Parameters
        ----------
            phy_id : int
                roi_id number for the region to be processed.

        """

        config = self.config

        shp = config["naipqq_query"]
        results_dir = config["results_dir"]
        id_field = config["roi_id"]

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

    def batch_clip_mosaic(self, pid):
        """
        Clips the mosaic to the ROI extent

        Parameters
        ----------
            phy_id : int
                roi_id number for the region to be processed.
        """

        config = self.config

        shp = config["roi_shp"]
        results_dir = config["results_dir"]
        proj = config["projection"]
        id_field = config["roi_id"]

        region_dir = '%s/%s' % (results_dir, pid)
        dir_path = '%s/Outputs' % (region_dir)
        input_raster_name = 'mosaic_%s.tif' % pid
        in_raster = '%s/%s' % (dir_path, input_raster_name)
        out_raster = '%s/clipped_%s' % (dir_path, input_raster_name)

        where = "%s = %d" % (id_field, pid)

        warp = gdal.Warp(out_raster, in_raster, xRes=1, yRes=1,
                         cutlineDSName=shp,
                         cutlineWhere=where, cropToCutline=True,
                         srcNodata='3', dstNodata='3',
                         outputType=gdal.GDT_Byte, creationOptions=["NBITS=2"],
                         dstSRS=proj)

    def batch_naip(self, pid, index, alg, smoothing=True,
                   class_parameters=None):
        """
        This function is a wrapper function run every step to make a canopy dataset.

        Parameters
        ----------
            phy_id : int
                roi_id number for the region to be processed.
            index : str, default="ARVI"
                Which vegetation index to compute with rindcalc
            alg: str
                Which classifiation algorithm to use
                "RF": Random Forests, "ET": Extra Trees
            smoothing : bool
                Whether or not to apply a 3x3 median filter
            class_parameters : dict
                Parameters to apply to classification

                Random Forests :

                {"n_estimators": 100, "criterion": 'gini', "max_depth": None,
                 "min_samples_split": 2, "min_samples_leaf": 1,
                 "min_weight_fraction_leaf": 0.0, "max_features": 'auto',
                 "max_leaf_nodes": None, "min_impurity_decrease": 0.0,
                 "min_impurity_split": None, "bootstrap": True,
                 "oob_score": False, "n_jobs": None, "random_state": None,
                 "verbose": 0, "warm_start": False, "class_weight": None,
                 "ccp_alpha": 0.0, "max_samples": None}

                Extra Trees :

                {"n_estimators": 100, "criterion": 'gini', "max_depth": None,
                 "min_samples_split": 2, "min_samples_leaf": 1,
                 "min_weight_fraction_leaf": 0.0, "max_features": 'auto',
                 "max_leaf_nodes": None, "min_impurity_decrease": 0.0,
                 "min_impurity_split": None, "bootstrap": False,
                 "oob_score": False, "n_jobs": None, "random_state": None,
                 "verbose": 0, "warm_start": False, "class_weight": None,
                 "ccp_alpha": 0.0, "max_samples": None}
        """
        self.batch_index(pid, index)
        if alg == "RF":
            self.batch_rf_class(pid, smoothing, class_parameters)
        if alg == "ET":
            self.batch_et_class(pid, smoothing, class_parameters)
        else:
            print("Enter either 'RF' or 'ET'")
        self.batch_clip_reproject(pid)
        self.batch_mosaic(pid)
        self.batch_clip_mosaic(pid)
        print('Finished')
