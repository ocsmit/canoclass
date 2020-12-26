import numpy as np
from osgeo import gdal
from utils import neighbors
from joblib import Parallel, delayed

class texture:

    def __init__(self, in_arr, threads, method='single', d=1):
        self.__d = d
        self.__threads = threads
        if method == 'multi':
            if type(in_arr) != list:
                raise IOError("Not list of arrays")
            vector_list = [np.array(self.__vector_variance(arr))
                            for arr in in_arr]
            self.vector = sum(vector_list) / len(vector_list)
        else:
            self.vector = np.array(self.__vector_variance(in_arr))
        return

    def __get_info(self, in_arr):
        self.in_arr = in_arr
        self.split_range = in_arr.shape[0] // self.__threads
        self.__remainder = in_arr.shape[0] % self.__threads
        self.__make_clusters()

    def __make_clusters(self):

        if self.__remainder !=0:
            self.data_clusters = [[i, i + self.split_range] for i in
                    range(0, self.in_arr.shape[0] - self.split_range,
                        self.split_range)]
            self.data_clusters[-1][1] += self.__remainder
        else:
            self.data_clusters = [[i, i + self.split_range] for i in
                    range(0, self.in_arr.shape[0], self.split_range)]

    def __cluster_dict(self):
        init_dict = {}
        for i in range(len(self.data_clusters)):
            init_dict.update({i: self.data_clusters[i]})
        self.data_clusters = init_dict

    def __vector_variance(self, arr):
        self.__get_info(arr)
        clusters = self.data_clusters
        results = Parallel(n_jobs=self.__threads)(delayed(self.get_var_vector)
                (arr, arr_split) for arr_split in clusters)
        vector = [item for sublist in results for item in sublist]
        return vector

    def get_var_vector(self, arr, arr_split):

        vector = []
        i1, i2 = arr_split
        for i in range(i1, i2):
            for j in range(arr.shape[1]):
                n = neighbors(arr, i, j, self.__d)
                variance = np.var(n)
                vector.append(variance)
        return vector





