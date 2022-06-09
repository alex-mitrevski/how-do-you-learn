import numpy as np
import matplotlib.pyplot as plt

def plot_confidence_interval(data: np.ndarray,
                             std_factor: float):
    """Plots a confidence interval for the given data.

    Keyword arguments:
    data: np.ndarray -- an m x n numpy array with n measurements and m repetitions
    std_factor: float -- multiplication factor for the standard deviation that determines
                         the confidence interval size (e.g, 1.96 for a 95% confidence interval)

    """
    items = np.array(list(range(0, data.shape[1])))
    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0)

    plt.plot(means, 'b-')
    plt.fill(np.concatenate([items, items[::-1]]),
             np.concatenate([means - std_factor * stds, (means + std_factor * stds)[::-1]]),
             alpha=.2, fc='g', ec='None')
    plt.show()
