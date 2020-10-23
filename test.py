import numpy as np
from mip_solver import solve_mip


def recover_pool_results(membership_matrix, pool_results, fpr, fnr, f):
    """
    :param membership_matrix: a numpy array, shape (num_pools * num_samples)
    :param pool_results: a numpy array, shape (num_pools * num_trials)
    :param fpr: a float between 0 and 1 indicating false positive rate
    :param fnr: a float between 0 and 1 indicating false negative rate
    :param f: a float between 0 and 1 indicating population infection rate
    :return: recovered infection vectors, shape (num_samples, num_trials)
    """
    num_pools, num_samples = membership_matrix.shape
    _, num_trials = pool_results.shape

    recovered_xs = np.empty((num_samples, num_trials))
    for trial in range(num_trials):
        if trial % 10 == 0:
            print("Starting trail %s ..." % trial)
        pool_result = pool_results[:, trial]
        solution, status = solve_mip(membership_matrix, pool_result, fpr, fnr, f)
        recovered_xs[:, trial] = solution

    return recovered_xs
