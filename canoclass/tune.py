import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import ExtraTreesClassifier
from canoclass.load_data import load_data
from canoclass.split_data import split_data


def tune(training_raster, training_fit_raster, start=10, step=10, stop=100):
    """
    Performs 5 fold cross validation to determine optimal parameters

    Args:
        training_raster: Rasterized training data
        training_fit_raster: Raster which data is drawn over

    """

    X, y = load_data(training_raster, training_fit_raster)

    X_train, X_test, y_train, y_test = split_data(training_raster,
                                                  training_fit_raster)

    n_estimators = [int(x) for x in np.linspace(start=start, stop=stop,
                    num=step)]
    min_samples_leaf = [int(x) for x in np.linspace(start=start, stop=stop,
                        num=step)]
    random_grid = {
        'n_estimators': n_estimators,
        'min_samples_leaf': min_samples_leaf
    }
    etc = ExtraTreesClassifier()
    clf = RandomizedSearchCV(etc, random_grid, random_state=0, verbose=3)
    clf.fit(X_train, y_train)

    print(clf.best_params_)
    return clf.cv_results_
