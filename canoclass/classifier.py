import os
import numpy as np
from osgeo import gdal
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.svm import SVC
from skimage import io
from .texture import texture
import multiprocessing as mp


class data_process:

    def __init__(self, arr_list, training_arr):

        self.arr_list = arr_list
        self.training_arr = training_arr.reshape(-1, 1)
        self.__get_texture()
        self.__stack_data()

    def __get_texture(self):
        self.__texture_band = texture(self.arr_list[3], mp.cpu_count(),
                method='single')

    def __stack_data(self):
        flattened_arrays = [band.reshape(-1, 1) for band in self.arr_list]
        flattened_arrays.append(self.__texture_band.vector.reshape(-1, 1))
        self.training_data_stack = np.vstack([flattened_arrays])
        self.predict_img = np.hstack(flattened_arrays)
        self.__make_fit_data(flattened_arrays)

    def __make_fit_data(self, flattened_arrays):

        self.y = self.training_arr[self.training_arr > 0]
        training_intersect = [arr[self.training_arr > 0] for arr in
                flattened_arrays]
        training_X = np.vstack(training_intersect)
        self.X= training_X.T


class classifier:

    def __init__(self, arr_list, training_data):
        self.data = data_process(arr_list, training_data)
        self.clf = PassiveAggressiveClassifier(n_jobs=-1)
        self.__process()

    def __process(self):

        rf = self.clf.fit(self.data.X, self.data.y)
        self.results = rf.predict(self.data.predict_img)

