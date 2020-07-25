from osgeo import gdal, ogr
import numpy as np
from scipy import ndimage
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score
from canoclass.split_data import split_data


def cross_val(training_raster, training_fit_raster, alg, folds=5,
              class_parameters=None):

    """
    Allows cross validation of training data while allowing the user to set the
    number of folds, choose the algorithm, and set the parameters with which to
    use for each algorithm. Training and test data is split automatically in a
    1/3 split.
    ---
    Args:
       training_raster : str, filepath
            Path of raster training data
       training_fit_raster : str, filepath
            Path of raster to fit training data to
       alg : ET or RF
            Which classifier algorithm to utilize "ET" or "RF"
       folds : int, default=5
            The number of folds to use in the cross validation
       class_parameters : dict
            Parameters to apply to classification

                * Random Forests ::
                {"n_estimators": 100, "criterion": 'gini', "max_depth": None,
                 "min_samples_split": 2, "min_samples_leaf": 1,
                 "min_weight_fraction_leaf": 0.0, "max_features": 'auto',
                 "max_leaf_nodes": None, "min_impurity_decrease": 0.0,
                 "min_impurity_split": None, "bootstrap": True,
                 "oob_score": False, "n_jobs": None, "random_state": None,
                 "verbose": 0, "warm_start": False, "class_weight": None,
                 "ccp_alpha": 0.0, "max_samples": None}

                 * Extra Trees ::
                {"n_estimators": 100, "criterion": 'gini', "max_depth": None,
                 "min_samples_split": 2, "min_samples_leaf": 1,
                 "min_weight_fraction_leaf": 0.0, "max_features": 'auto',
                 "max_leaf_nodes": None, "min_impurity_decrease": 0.0,
                 "min_impurity_split": None, "bootstrap": False,
                 "oob_score": False, "n_jobs": None, "random_state": None,
                 "verbose": 0, "warm_start": False, "class_weight": None,
                 "ccp_alpha": 0.0, "max_samples": None}
     """

    X_train, X_test, y_train, y_test = split_data(training_raster,
                                                  training_fit_raster)

    if alg == "RF":
        if class_parameters is None:
            parameters = {"n_estimators": 100, "criterion": 'gini', "max_depth": None,
                          "min_samples_split": 2, "min_samples_leaf": 1,
                          "min_weight_fraction_leaf": 0.0, "max_features": 'auto',
                          "max_leaf_nodes": None, "min_impurity_decrease": 0.0,
                          "min_impurity_split": None, "bootstrap": True,
                          "oob_score": False, "n_jobs": None, "random_state": None,
                          "verbose": 0, "warm_start": False, "class_weight": None,
                          "ccp_alpha": 0.0, "max_samples": None}
            clf = RandomForestClassifier(**parameters)
        else:
            parameters = class_parameters
            clf = RandomForestClassifier(**parameters)

    if alg == "ET":
        if class_parameters is None:
            parameters = {"n_estimators": 100, "criterion": 'gini', "max_depth": None,
                          "min_samples_split": 2, "min_samples_leaf": 1,
                          "min_weight_fraction_leaf": 0.0, "max_features": 'auto',
                          "max_leaf_nodes": None, "min_impurity_decrease": 0.0,
                          "min_impurity_split": None, "bootstrap": False,
                          "oob_score": False, "n_jobs": None, "random_state": None,
                          "verbose": 0, "warm_start": False, "class_weight": None,
                          "ccp_alpha": 0.0, "max_samples": None}
            clf = ExtraTreesClassifier(**parameters)
        else:
            parameters = class_parameters
            clf = ExtraTreesClassifier(**parameters)

    scores = cross_val_score(clf, X_test, y_test, cv=folds)

    return scores

