from sklearn.model_selection import train_test_split
from canoclass.load_data import load_data


def split_data(training_raster, training_fit_raster):
    """
    Split data into training and testing data

    Args:
        training_raster: Rasterized training data
        training_fit_raster: Raster which data is drawn over

    """

    X, y = load_data(training_raster, training_fit_raster)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    return X_train, X_test, y_train, y_test
