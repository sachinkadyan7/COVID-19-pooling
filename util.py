import numpy as np
import random
import os
import math


def check_inputs(fpr, fnr, f):
    """
    Get valid inputs (concerning zeros).
    :param fpr: false positive rate.
    :param fnr: false negative rate.
    :param f: population infection rate.
    :return: valid inputs.
    """
    assert f != 0, "Please input a non-zero infection rate."
    if fpr == 0:
        fpr = np.nextafter(0, 1)
    if fnr == 0:
        fnr = np.nextafter(0, 1)
    return fpr, fnr, f


def simulate_x(n, f, num_trials=1000, filepath=None):
    """
    Code to generate infection vector for testing.
    :param n: integer, number of samples.
    :param f: population infection rate.
    :param num_trials: integer, number of trials, default to 100 infection vectors.
    :param filepath: path of the simulated vector, default in '/data/' folder.
    :return: None, saves the vectors to a csv file.
    """
    if not os.path.exists("./data/"):
        os.mkdir("./data/")

    rows = n
    cols = num_trials

    if filepath is None:
        filepath = "./data/n%s-f%.4f-numTrials%s.csv" % (n, f, num_trials)

    xs = np.empty(shape=(n, num_trials))

    for r in range(rows):
        for c in range(cols):
            xs[r][c] = int(np.random.binomial(1, f, 1))  # binomial(n,p,trials)

    print("On average, %.2f positives in each trail." % np.average(xs.sum(0)))
    np.savetxt(filepath, xs, delimiter=',')  # convert to csv


def simulate_pool_results(xs, M, fpr, fnr):
    """
    Simulate pool results with false positive rate fpr and false negative rate fnr.
    :param xs: infection vectors.
    :param M: membership matrix.
    :param fpr: false positive rate.
    :param fnr: false negative rate.
    :return: simulated pool results.
    """
    num_pools, num_samples = M.shape
    _, num_trials = xs.shape

    sgn_Mxs = np.sign(M @ xs)
    pool_results = np.zeros(sgn_Mxs.shape)
    fps = np.zeros(sgn_Mxs.shape)  # false positives
    fns = np.zeros(sgn_Mxs.shape)  # false negatives

    for j in range(num_trials):
        for i in range(num_pools):
            r = random.uniform(0, 1)
            if sgn_Mxs[i, j] == 0 and r < fpr:
                pool_results[i, j] = 1
                fps[i, j] = 1
            elif sgn_Mxs[i, j] == 1 and r < fnr:
                pool_results[i, j] = 0
                fns[i, j] = 1
            else:
                pool_results[i, j] = sgn_Mxs[i, j]

    assert np.all(pool_results == sgn_Mxs + fps - fns)

    return pool_results, fps, fns


def divisor_generator(n):
    """
    Get the divisors of n.
    :param n: a natural number.
    :return: the divisors of n.
    """
    large_divisors = []
    for i in range(1, int(math.sqrt(n) + 1)):
        if n % i == 0:
            yield int(i)
            if i*i != n:
                large_divisors.append(n / i)
    for divisor in reversed(large_divisors):
        yield int(divisor)


def get_Ts(n, col_weight):
    """
    Return the numbers of pools for n samples with column weight col_weight.
    :param n: number of samples.
    :param col_weight: column weight.
    :return: the numbers of pools.
    """
    divisors = list(divisor_generator(n))
    return [divisor for divisor in divisors if divisor > col_weight][:-1]
