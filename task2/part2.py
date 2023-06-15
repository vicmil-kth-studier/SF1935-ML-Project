"""
Generate and plot data points using the model defined in Eq. 37 over the 2D domain
of the input space where x = [x1, x2] ∈ [−1, −0.95, . . . , 0.95, 1] × [−1, −0.95, . . . , 0.95, 1]
(use different fixed values of data noise σ, say, σ

sigma^2 ∈ {0.2, 0.4, 0.6}).
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import scipy
import random
import math
from typing import List

random.seed(4)
np.random.seed(4)

w0_param = 0
w1_param = 1.5
w2_param = -0.8

def generate_samples(noice_variance = 0.2):
    noice_mean = 0
    samples_x1 = list()
    samples_x2 = list()
    samples_t = list()
    lower = -1.0
    upper = 1.0
    length = (upper - lower) / 0.05 + 1
    for x1 in [lower + x*(upper-lower)/(length-1) for x in range(int(length))]:
        for x2 in [lower + x*(upper-lower)/(length-1) for x in range(int(length))]:
            samples_x1.append(x1)
            samples_x2.append(x2)
            noice = np.random.normal(noice_mean, noice_variance, 1)[0]
            t = w0_param + w1_param * x1 + w2_param * x2
            t = t + noice
            samples_t.append(t)

    return samples_x1, samples_x2, samples_t

def randomize_sample_order(samples_x1, samples_x2, samples_t):
    ret_samples_x1 = list()
    ret_samples_x2 = list()
    ret_samples_t = list()
    indexes = list()
    for i in range(len(samples_x1)):
        indexes.append(i)

    random.shuffle(indexes)
    for i in range(len(samples_x1)):
        index = indexes[i]
        ret_samples_x1.append(samples_x1[index])
        ret_samples_x2.append(samples_x2[index])
        ret_samples_t.append(samples_t[index])

    return ret_samples_x1, ret_samples_x2, ret_samples_t

# Generate sample data
samples_x1, samples_x2, samples_t = generate_samples(noice_variance=0.2)
samples_x1, samples_x2, samples_t = randomize_sample_order(samples_x1, samples_x2, samples_t)

training_samples_x1 = list()
training_samples_x2 = list()
training_samples_t = list()
test_samples_x1 = list()
test_samples_x2 = list()
test_samples_t = list()
for i in range(len(samples_x1)):
    if samples_x1[i] > 0.3 and samples_x2[i] > 0.3:
        test_samples_x1.append(samples_x1[i])
        test_samples_x2.append(samples_x2[i])
        test_samples_t.append(samples_t[i])
    else:
        training_samples_x1.append(samples_x1[i])
        training_samples_x2.append(samples_x2[i])
        training_samples_t.append(samples_t[i])

# Plot the data in the 3d space since it depends on both x1 and x2
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
for i in range(len(samples_x1)):
    color_r = (samples_x1[i] - min(samples_x1)) / (max(samples_x1) - min(samples_x1))
    color_g = (samples_x2[i] - min(samples_x2)) / (max(samples_x2) - min(samples_x2))
    color_b = (samples_t[i] - min(samples_t)) / (max(samples_t) - min(samples_t))
    ax.scatter(samples_x1[i], samples_x2[i], samples_t[i], c = (color_r, color_b, color_g), s = 5)
plt.show(block=True)
print("Done!")
