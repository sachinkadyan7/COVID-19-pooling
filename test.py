import numpy as np
from mip_solver import solve_mip
import pandas as pd


def recover_pool_results(membership_matrix, pool_results, fpr, fnr, f, verbose=False):
    """
    :param membership_matrix: a numpy array, shape (num_pools * num_samples)
    :param pool_results: a numpy array, shape (num_pools * num_trials)
    :param fpr: a float between 0 and 1 indicating false positive rate
    :param fnr: a float between 0 and 1 indicating false negative rate
    :param f: a float between 0 and 1 indicating population infection rate
    :param verbose: set verbose to True to surpress print statements. 
    :return: recovered infection vectors, shape (num_samples, num_trials)
    """
    num_pools, num_samples = membership_matrix.shape
    _, num_trials = pool_results.shape

    recovered_xs = np.empty((num_samples, num_trials))
    recovered_false_ps = np.empty((num_pools, num_trials))
    recovered_false_ns = np.empty((num_pools, num_trials))
    for trial in range(num_trials):
        if trial % 10 == 0 and not verbose:
            print("Starting trail %s ..." % trial)
        pool_result = pool_results[:, trial]
        recovered_x, recovered_false_p, recovered_false_n, status = solve_mip(membership_matrix, pool_result, fpr, fnr, f)
        recovered_xs[:, trial] = recovered_x
        recovered_false_ps[:, trial] = recovered_false_p
        recovered_false_ns[:, trial] = recovered_false_n

    return recovered_xs, recovered_false_ps, recovered_false_ns


def simulate_x(num_samples, num_trials, f, filename):
    """
    Code to generate infection vector for testing.
    By Sunny.
    :param num_samples: integer, number of samples.
    :param num_trials: integer, number of trials.
    :param f: population infection rate.
    :param filename: filename, file can be found in ./tests/data/ folder.
    :return: None, saves the vectors to a csv file.
    """
    rows = num_samples  # 384
    cols = num_trials  # 100

    xs = []
    for row in range(rows):
        xs += [[7] * cols]  # Weird initial number to make sure every cell gets touched by next loop

    for r in range(rows):
        for c in range(cols):
            xs[r][c] = int(np.random.binomial(1, f, 1))  # binomial(n,p,trials)

    final = np.array(xs)  # convert to numpy array (So it can be used as input to Sachin's solver)
    print(final)
    print(final.shape)
    df = pd.DataFrame(final)  # convert to pandas dataframe
    df.to_csv("./tests/data/" + filename)  # convert to csv
