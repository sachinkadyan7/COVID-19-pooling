import numpy as np
import random
import os


def check_inputs(fpr, fnr, f):
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
    :param filepath: path of the simulated vector, default in '/data/' folder.
    :param num_trials: integer, number of trials, default to 100 infection vectors.
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


def simulate_pool_results(xs, membership_matrix, fpr, fnr):
    num_pools, num_samples = membership_matrix.shape
    _, num_trials = xs.shape

    sgn_Mxs = np.sign(membership_matrix @ xs)
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

