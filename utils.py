from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np
import matplotlib.pyplot as plt

def get_data(fetch_data, test_size=.25):
    '''

    Simple utility to fetch data

    Parameters
    ----------
    fetch_data: callable
        A function for fetching data
    test_size: float
        Test set size

    Returns
    -------
    X_train: numpy.ndarray
       Train set features
    X_test: numpy.ndarray
       Test set features
    y_train: numpy.ndarray
       Train set targets
    y_test: numpy.ndarray
       Test set targets

    '''
    bunch = fetch_data()
    X = bunch.data
    y = bunch.target
    X, y = shuffle(X, y)
    return train_test_split(X, y, test_size=test_size)


def plot_approximation_line(W, b, X):
    linspace = np.linspace(np.min(X), np.max(X))
    line = linspace * W  + b
    plt.plot(line)



if __name__ == '__main__':
    from sklearn.datasets import fetch_california_housing

    X_train, X_test, y_train, y_test = get_data(fetch_california_housing)

    import pdb
    pdb.set_trace()