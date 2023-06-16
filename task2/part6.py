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

random.seed(6)
np.random.seed(6)

w0_param = 0
w1_param = 1.5
w2_param = -0.8

class Result:
    def __init__(self):
        self.test_mean = 0
        self.test_variance = 0
        self.test_msq = 0
        self.training_mean = 0
        self.training_variance = 0
        self.training_msq = 0
        self.training_on_test_msq = 0
        self.training_on_test_variance = 0
        self.training_samples_x1 = list()
        self.training_samples_x2 = list()
        self.training_samples_t = list()
        self.test_samples_x1 = list()
        self.test_samples_x2 = list()
        self.test_samples_t = list()
        self.sample_bias = 0


def task6(noice_variance = 0.4, alpha = 1.5) -> Result:
    beta = 1 / noice_variance
    result = Result()
    print("variance ", noice_variance)
    print("alpha ", alpha)

    def generate_samples():
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
                t = w0_param + w1_param * x1 + w2_param * x2
                t = t + noice
                samples_t.append(t)

        return samples_x1, samples_x2, samples_t

    def get_sample_variance_estimator(samples_x1, samples_x2, samples_t):
        sum = 0
        for i in range(len(samples_x1)):
            x1 = samples_x1[i]
            x2 = samples_x2[i]
            t = samples_t[i]
            t_val = w0_param + w1_param * x1 + w2_param * x2
            sum += pow(t - t_val, 2)

        return sum / (len(samples_x1))

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
    #print("sample variance ", get_sample_variance_estimator(samples_x1, samples_x2, samples_t))
    samples_x1, samples_x2, samples_t = randomize_sample_order(samples_x1, samples_x2, samples_t)

    for i in range(len(samples_x1)):
        if samples_x1[i] > 0.3 and samples_x2[i] > 0.3:
            result.test_samples_x1.append(samples_x1[i])
            result.test_samples_x2.append(samples_x2[i])
            result.test_samples_t.append(samples_t[i])
        else:
            result.training_samples_x1.append(samples_x1[i])
            result.training_samples_x2.append(samples_x2[i])
            result.training_samples_t.append(samples_t[i])

    """result.training_mean, result.training_variance = get_predicted_mean_and_variance(result.training_samples_x1, result.training_samples_x2, result.training_samples_t, [0.3, 0.3])
    result.test_mean, result.test_variance = get_predicted_mean_and_variance(result.test_samples_x1, result.test_samples_x2, result.test_samples_t, [0.3, 0.3])
    """

    #print("training mean, variance", result.training_mean, result.training_variance)
    #print("test mean, variance", result.test_mean, result.test_variance)

    # Plot the prob function
    """x_vals = list()
    y_vals = list()
    i = -2
    while i <= 2:
        i = i + 0.01
        x_vals.append(i)
        p = normal_pdf(mean=result.training_mean, variance=result.training_variance, data_point_t=i)
        y_vals.append(p)"""

    #plt.figure()
    #plt.scatter(x_vals, y_vals, c="blue")

    # Plot the prob function
    """x_vals = list()
    y_vals = list()
    i = -2
    while i <= 2:
        i = i + 0.01
        x_vals.append(i)
        p = normal_pdf(mean=result.test_mean, variance=result.test_variance, data_point_t=i)
        y_vals.append(p)"""

    #plt.scatter(x_vals, y_vals, c="red")

    #plt.ylabel('p')
    #plt.xlabel('t')

    # Calculate the mean square error of the predictions
    msq_sum = 0
    variance_sum = 0
    for i in range(len(result.test_samples_t)):
        x1 = result.test_samples_x1[i]
        x2 = result.test_samples_x2[i]
        t = result.test_samples_t[i]
        predicted_t, variance = get_predicted_mean_and_variance(result.test_samples_x1, result.test_samples_x2, result.test_samples_t, [x1, x2])
        err = t - predicted_t
        msq_sum += err * err
        variance_sum += variance

    result.test_msq = msq_sum / (len(result.test_samples_t))
    result.test_variance = variance_sum / (len(result.test_samples_t))
    #print("test msq", result.test_msq)


    # Calculate the mean square error of the predictions
    msq_sum = 0
    variance_sum = 0
    for i in range(len(result.training_samples_t)):
        x1 = result.training_samples_x1[i]
        x2 = result.training_samples_x2[i]
        t = result.training_samples_t[i]
        predicted_t, variance = get_predicted_mean_and_variance(result.training_samples_x1, result.training_samples_x2, result.training_samples_t, [x1, x2])
        err = t - predicted_t
        msq_sum += err * err
        variance_sum += variance

    result.training_msq = msq_sum / (len(result.training_samples_t))
    result.training_variance = variance_sum / (len(result.training_samples_t))
    #print("training on test msq", result.training_on_test_msq)


    # Calculate the mean square error of the predictions
    msq_sum = 0
    variance_sum = 0
    for i in range(len(result.test_samples_t)):
        x1 = result.test_samples_x1[i]
        x2 = result.test_samples_x2[i]
        t = result.test_samples_t[i]
        predicted_t, variance = get_predicted_mean_and_variance(result.training_samples_x1, result.training_samples_x2, result.training_samples_t, [x1, x2])
        err = t - predicted_t
        msq_sum += err * err
        variance_sum += variance

    result.training_on_test_msq = msq_sum / (len(result.test_samples_t))
    result.training_on_test_variance = variance_sum / (len(result.test_samples_t))
    #print("training on test msq", result.training_msq)

    # Plot the predictions and the actual data
    """x1_vals = list()
    x2_vals = list()
    y_vals = list()
    for i in range(200):
        x1 = result.training_samples_x1[i]
        x2 = result.training_samples_x2[i]
        t = result.training_samples_t[i]
        predicted_t, _ = get_predicted_mean_and_variance(result.training_samples_x1, result.training_samples_x2, result.training_samples_t, [x1, x2])
        x1_vals.append(x1)
        x2_vals.append(x2)
        y_vals.append(predicted_t)"""

    #fig = plt.figure()
    #ax = fig.add_subplot(projection='3d')
    #ax.scatter(training_samples_x1, training_samples_x2, training_samples_t, c = "green", s = 5)
    #ax.scatter(x1_vals, x2_vals, y_vals, c = "red", s = 5)
    #plt.show(block=True)
    result.sample_bias = get_sample_variance_estimator(result.training_samples_x1, result.training_samples_x2, result.training_samples_t)

    return result


def get_mean(data: List[float]):
    sum = 0
    for i in data:
        sum += i

    return sum / len(data)

def get_variance(data: List[float]):
    mean = get_mean(data)
    sum = 0
    for i in data:
        sum += (i - mean) * (i - mean)

    return sum / (len(data) - 1)


results: List[Result] = list()
for i in range(0, 100):
    result = task6(alpha=0.7, noice_variance=0.6)
    results.append(result)
    #print(result.__dict__)
    #print("hello")
    print(len(results))

keys = results[0].__dict__.keys()
for key in keys:
    #print(type(results[0].__dict__[key]))
    if(type(results[0].__dict__[key]) != float and type(results[0].__dict__[key]) != np.float64):
        #print("skipped ", key)
        continue
    data = list()
    for i in range(len(results)):
        data.append(results[i].__dict__[key])

    #print(data)
    print(key == "training_on_test_msq" or key == "training_variance", key, " mean", get_mean(data))
    print(key == "training_on_test_msq" or key == "training_variance", key, " standard deviation", pow(get_variance(data), 0.5))
