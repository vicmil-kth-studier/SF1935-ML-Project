"""
Set the prior distribution over w with α = 2 (or 1, 3) and visualise it. Although it is a
multivariate normal distribution (in 3D), make two-dimensional contour plot over the
w0 × w1 space. To calculate the values of the prior (Gaussian distribution), you could
simply use Python function for multivariate normal distribution or indirectly calculate
from the exp function using np.exp(D) (to compute the exponetial of all elements in a
matrix) and cdist(x1,x2) (to compute a distance matrix between two sets of vectors).
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import scipy
import random
from typing import List

random.seed(4)
np.random.seed(4)

def plot_contour_plot(x: List[List[float]], y: List[List[float]], z: List[List[float]], x_label="w0", y_label="w1"):
    # x is indexed by [x, y]
    # y is indexed by [y, x]
    # z is indexed by [x, y]
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    ax2.contourf(x, y, z)

def get_data_for_distribution(mean_w0, mean_w1, variance_matrix: List[List[float]]):
    w0, w1 = np.mgrid[-1:1:.01, -1:1:.01]
    pos = np.dstack((w0, w1))
    rv = multivariate_normal([mean_w0, mean_w1], variance_matrix)
    z = rv.pdf(pos)
    return w0, w1, z

def get_first_prior_distribution(alpha):
    variance = 1 / alpha
    variance_matrix = [[variance, 0], [0, variance]]
    mean = 0
    return get_data_for_distribution(mean, mean, variance_matrix)

w0, w1, z = get_first_prior_distribution(2)
plot_contour_plot(w0, w1, z)
plt.show(block=True)