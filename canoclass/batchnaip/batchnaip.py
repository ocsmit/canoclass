def batch_naip(pid, index, alg, smoothing=True, class_parameters=None):
    """
    This function is a wrapper function run every step to make a canopy dataset.
    Args:
        phy_id: int ::  Physio Id for the region to be processed.
        index: str :: Vegetation index to calculate with Rindcalc
        alg: str :: Which classifiation algorithm to use
                Options :: "RF": Random Forests, "ET": Extra Trees
        smoothing :: boolean : whether or not to apply a 3x3 median filter
        class_parameters :: dict : Parameters to apply to classification
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
    batchIndex(pid, index)
    if alg == "RF":
        batch_rf_class(pid, smoothing, class_parameters)
    if alg == "ET":
        batch_et_class(pid, smoothing, class_parameters)
    else:
        print("Enter either 'RF' or 'ET'")
    batch_et_class(pid)
    batch_clip_reproject(pid)
    batch_mosaic(pid)
    clip_mosaic(pid)
    print('Finished')
