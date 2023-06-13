"""
For the corresponding values of σ^2 and different values of the uncertainty parameter
of the Gaussian prior over the weight parameters (α ∈ {0.7, 1.5, 3.0}), please perform
Bayesian linear regression.
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
noice_variance = 0.4
beta = 1 / noice_variance
alpha = 1.5

print("variance ", noice_variance)
print("alpha ", alpha)

def generate_samples():
    w0 = 0.25
    w1 = 0.89
    w2 = -0.52
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
            standard_deviation = pow(noice_variance, 0.5)
            noice = np.random.normal(noice_mean, standard_deviation, 1)[0]
            t = w0 + w1 * x1 + w2 * x2
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

def get_mean_sqaure_error(samples_x1, samples_x2, samples_t, w: List[float]):
    # Calculate the mean square error
    sum = 0
    for i in range(len(samples_t)):
        x1 = samples_x1[i]
        x2 = samples_x2[i]
        t = samples_t[i]
        predicted_t = w[0] + w[1] * x1 + w[2] * x2
        err = t - predicted_t
        sum += err * err
    
    mean_square_err = sum / len(samples_t)
    return mean_square_err

def normal_pdf(mean, variance, data_point_t):
    e_factor = -0.5 * ((data_point_t - mean)/(variance))*((data_point_t - mean)/(variance))
    t_prob = (1 / (pow(2*math.pi, 0.5) * variance)) * pow(math.e, e_factor)
    return t_prob

def get_predicted_mean_and_variance(samples_x1, samples_x2, samples_t, x: List[float]):
    t = list()
    Phi = list()
    sample_count = len(samples_t)
    for i in range(0, sample_count):
        t.append(samples_t[i])
        Phi.append([1, samples_x1[i], samples_x2[i]])
    Phi = np.array(Phi)
    t = np.array(t)

    # Calculate the prediction from the training data:
    x_ext = [1, x[0], x[1]]
    x = np.array(x)
    x_ext = np.array(x_ext)

    S_n_inv = alpha * np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    S_n_inv = S_n_inv + beta * Phi.transpose().dot(Phi)
    S_n = np.linalg.inv(S_n_inv)
    mean = Phi.transpose().dot(t)
    mean = beta * S_n.dot(mean)
    mean = mean.transpose().dot(x_ext)
    variance = S_n.dot(x_ext)
    variance = x_ext.transpose().dot(variance)
    variance = 1 / beta + variance

    return mean, variance

# Generate sample data
samples_x1, samples_x2, samples_t = generate_samples()
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

mean1, variance1 = get_predicted_mean_and_variance(training_samples_x1, training_samples_x2, training_samples_t, [0.3, 0.3])
mean2, variance2 = get_predicted_mean_and_variance(test_samples_x1, test_samples_x2, test_samples_t, [0.3, 0.3])

print("training mean, variance", mean1, variance1)
print("test mean, variance", mean2, variance2)

# Plot the prob function
x_vals = list()
y_vals = list()
i = -2
while i <= 2:
    i = i + 0.01
    x_vals.append(i)
    p = normal_pdf(mean=mean1, variance=variance1, data_point_t=i)
    y_vals.append(p)

plt.figure()
plt.scatter(x_vals, y_vals, c="blue")

# Plot the prob function
x_vals = list()
y_vals = list()
i = -2
while i <= 2:
    i = i + 0.01
    x_vals.append(i)
    p = normal_pdf(mean=mean2, variance=variance2, data_point_t=i)
    y_vals.append(p)

plt.scatter(x_vals, y_vals, c="red")

plt.ylabel('p')
plt.xlabel('t')

# Calculate the mean square error of the predictions
sum = 0
for i in range(len(test_samples_t)):
    x1 = test_samples_x1[i]
    x2 = test_samples_x2[i]
    t = test_samples_t[i]
    predicted_t, _ = get_predicted_mean_and_variance(test_samples_x1, test_samples_x2, test_samples_t, [x1, x2])
    err = t - predicted_t
    sum += err * err

mean_square_err = sum / len(test_samples_t)
print("test msq", mean_square_err)


# Calculate the mean square error of the predictions
sum = 0
for i in range(len(training_samples_t)):
    x1 = training_samples_x1[i]
    x2 = training_samples_x2[i]
    t = training_samples_t[i]
    predicted_t, _ = get_predicted_mean_and_variance(training_samples_x1, training_samples_x2, training_samples_t, [x1, x2])
    err = t - predicted_t
    sum += err * err

mean_square_err = sum / len(training_samples_t)
print("training msq", mean_square_err)


# Calculate the mean square error of the predictions
sum = 0
for i in range(len(test_samples_t)):
    x1 = test_samples_x1[i]
    x2 = test_samples_x2[i]
    t = test_samples_t[i]
    predicted_t, _ = get_predicted_mean_and_variance(training_samples_x1, training_samples_x2, training_samples_t, [x1, x2])
    err = t - predicted_t
    sum += err * err

mean_square_err = sum / len(test_samples_t)
print("training on test msq", mean_square_err)

# Plot the predictions and the actual data
x1_vals = list()
x2_vals = list()
y_vals = list()
for i in range(200):
    x1 = training_samples_x1[i]
    x2 = training_samples_x2[i]
    t = training_samples_t[i]
    predicted_t, _ = get_predicted_mean_and_variance(training_samples_x1, training_samples_x2, training_samples_t, [x1, x2])
    x1_vals.append(x1)
    x2_vals.append(x2)
    y_vals.append(predicted_t)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(training_samples_x1, training_samples_x2, training_samples_t, c = "green", s = 5)
ax.scatter(x1_vals, x2_vals, y_vals, c = "red", s = 5)
plt.show(block=True)
