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

w0_param = 0
w1_param = 1.5
w2_param = -0.8

class Result:
    def __init__(self):
        self.max_lik_msq = 0
        self.bay_msq = 0
        self.training_samples_x1 = list()
        self.training_samples_x2 = list()
        self.training_samples_t = list()
        self.test_samples_x1 = list()
        self.test_samples_x2 = list()
        self.test_samples_t = list()

def task4(noice_variance = 0.4, alpha = 1.5) -> Result:
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

    def get_mean_sqaure_error(test_samples_x1, test_samples_x2, test_samples_t, w: List[float]):
        # Calculate the mean square error on the test data
        sum = 0
        for i in range(len(test_samples_t)):
            x1 = test_samples_x1[i]
            x2 = test_samples_x2[i]
            t = test_samples_t[i]
            predicted_t = w[0] + w[1] * x1 + w[2] * x2
            err = t - predicted_t
            sum += err * err
        
        mean_square_err = sum / len(test_samples_t)
        return mean_square_err

    # Generate sample data
    samples_x1, samples_x2, samples_t = generate_samples()
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

    # Fit the model using maximum likelihood
    # Get the X_ext and use that to get the maximum likelihood
    t = list()
    Phi = list()
    sample_count = len(result.training_samples_t)
    for i in range(0, sample_count):
        t.append(result.training_samples_t[i])
        Phi.append([1, result.training_samples_x1[i], result.training_samples_x2[i]])
    Phi = np.array(Phi)
    t = np.array(t)

    # Perform bayesian linear regression (same as in task 1)
    ident_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    PhiDotPhi = Phi.transpose().dot(Phi)
    variance_inv = alpha * ident_matrix + beta * PhiDotPhi
    variance = np.linalg.inv(variance_inv)
    mean = Phi.transpose().dot(t)
    mean = variance.dot(mean.transpose())
    mean = (beta * mean).transpose()

    w_ml_bay = mean

    result.bay_msq = get_mean_sqaure_error(result.test_samples_x1, result.test_samples_x2, result.test_samples_t, w_ml_bay)
    #print(f"mean square err: {mean_square_err_bay}")
    #print(f"bay sqrt mean square err: {result.bay_msq}")


    # Perform maximum likelihood
    w_ml1 = np.linalg.inv(Phi.transpose().dot(Phi))
    w_ml2 = Phi.transpose().dot(t)
    w_ml_freq = w_ml1.dot(w_ml2)
    #print(w_ml_freq)

    result.max_lik_msq = get_mean_sqaure_error(result.test_samples_x1, result.test_samples_x2, result.test_samples_t, w_ml_freq)
    #print(f"mean square err: {mean_square_err_freq}")
    #print(f"freq sqrt mean square err: {result.max_lik_msq}")

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
    result = task4(alpha=3.0, noice_variance=0.6)
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

