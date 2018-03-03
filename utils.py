from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

def get_data(fetch_data, test_size=.25, features=0):
    '''

    Simple utility to fetch data

    :param fetch_data:
    :param test_size:
    :return:
    '''
    bunch = fetch_data()
    X = bunch.data#[:, features].reshape(-1, 1)
    y = bunch.target
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