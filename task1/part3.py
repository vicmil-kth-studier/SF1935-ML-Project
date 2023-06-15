"""
Go back to the generated data (samples) and pick a single data point (x,t). Then cal-
culate and plot the likelihood for this single data point across all w in your parameter
space (w0 × w1) - see Eq. 16 (for a single data point it is a univariate case og Gaussian
likelihood). Next visualise the posterior distribution over w. You can obtain the "em-
pirical" posterior from the prior and the likelihood by multiplying them and ignoring the
normalising constant (sec. 1.4). It is important that the prior in the next iteration cor-
responds to the posterior calculated in the previous iteration (in the very first iteration
we have no posterior so the prior is initialised in step 1).
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
plt.rcParams.update({'font.size': 16})
w0_param = -1.5
w1_param = 0.5


def plot_contour_plot(x: List[List[float]], y: List[List[float]], z: List[List[float]], x_label="w0", y_label="w1"):
    # x is indexed by [ind_x, ind_y] and reflects the x value at that position
    # y is indexed by [ind_x, ind_y] and reflects the y value at that position
    # z is indexed by [ind_x, ind_y] and reflects the z value at that position
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    plt.subplots_adjust(left=0.20, top=0.95, right=0.95, bottom=0.15)
    plt.xlabel(x_label, fontsize=20)
    plt.ylabel(y_label, fontsize=20)
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

def generate_samples():
    noice_mean = 0
    noice_variance = 0.2
    samples_x = list()
    samples_t = list()
    lower = -2.0
    upper = 2.0
    length = (upper - lower) / 0.01 + 1
    for x in [lower + x*(upper-lower)/(length-1) for x in range(int(length))]:
        samples_x.append(x)
        noice = np.random.normal(noice_mean, noice_variance, 1)[0]
        t = w0_param + w1_param * x
        t = t + noice
        samples_t.append(t)

    return samples_x, samples_t

def randomize_sample_order(samples_x, samples_t):
    ret_samples_x = list()
    ret_samples_t = list()
    indexes = list()
    for i in range(len(samples_x)):
        indexes.append(i)

    random.shuffle(indexes)
    for i in range(len(samples_x)):
        index = indexes[i]
        ret_samples_x.append(samples_x[index])
        ret_samples_t.append(samples_t[index])

    return ret_samples_x, ret_samples_t

def normal_pdf(mean, variance, data_point_t):
    e_factor = -0.5 * ((data_point_t - mean)/(variance))*((data_point_t - mean)/(variance))
    t_prob = (1 / (pow(2*math.pi, 0.5) * variance)) * pow(math.e, e_factor)
    return t_prob


alpha = 2
beta = 2

# Generate the samples
samples_x, samples_t = generate_samples()
samples_x, samples_t = randomize_sample_order(samples_x, samples_t)

# Choose the first as our data point
data_point_x = samples_x[0]
data_point_t = samples_t[0]

# Get the likelihood distribution
w0_grid, w1_grid = np.mgrid[-1:1:.01, -1:1:.01]
z, _ = np.mgrid[-1:1:.01, -1:1:.01]
for x_idx in range(len(w0_grid)):
    print(x_idx)
    for y_idx in range(len(w0_grid[x_idx])):
        w0 = w0_grid[x_idx][y_idx]
        w1 = w1_grid[x_idx][y_idx]
        mean = w0 + w1 * data_point_x
        variance = 1 / beta
        t_prob = normal_pdf(mean, variance, data_point_t)
        #t_prob2 = scipy.stats.norm(mean, variance).pdf(data_point_t)
        #err = abs(t_prob / t_prob2 - 1)
        #assert(err < 0.001)
        z[x_idx][y_idx] = t_prob

# Plot the likelihood distribution
plot_contour_plot(w0_grid, w1_grid, z)

# Get the posterior distribution
t = list()
Phi = list()
for i in range(0, 1):
    print(f"({samples_x[i]}, {samples_t[i]})")
    t.append(samples_t[i])
    Phi.append([1, samples_x[i]])
Phi = np.array(Phi)
ident_matrix = np.array([[1, 0], [0, 1]])
PhiDotPhi = Phi.transpose().dot(Phi)
variance_inv = alpha * ident_matrix + beta * PhiDotPhi
variance = np.linalg.inv(variance_inv)
mean = Phi.transpose().dot(t)
mean = variance.dot(mean.transpose())
mean = (beta * mean).transpose()


# Plot posterior
w0, w1, z = get_data_for_distribution(mean[0], mean[1], variance)
plot_contour_plot(w0, w1, z)
#plt.scatter([0.25], [0.89], c="red")

# Draw 5 samples from posterior
w0_samples, w1_samples = np.random.multivariate_normal(mean, variance, 5).T

# Plot the 5 posterior samples
fig3 = plt.figure()
for i in range(len(w0_samples)):
    x = [min(samples_x), max(samples_x)]
    y = [w0_samples[i] + x[0]*w1_samples[i], w0_samples[i] + x[1]*w1_samples[i]]
    plt.plot(x, y)

# Plot the sampled data
#plt.scatter(samples_x, samples_t, c="red", s=2)
plt.scatter(samples_x[:1], samples_t[:1], c="green", s=20)

plt.show(block=True)

# How do I handle multiple data points????

while True:
    pass
print("done!")
# Plot them
