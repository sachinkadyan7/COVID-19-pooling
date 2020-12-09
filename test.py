import numpy as np
from mip_solver import solve_mip
from util import check_inputs, simulate_pool_results
import scipy.io


def recover_pool_results(membership_matrix, pool_results, fpr, fnr, f, test=False):
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
        pool_result = pool_results[:, trial]
        recovered_x, recovered_false_p, recovered_false_n = solve_mip(membership_matrix, pool_result, fpr, fnr, f, test)
        recovered_xs[:, trial] = recovered_x
        recovered_false_ps[:, trial] = recovered_false_p
        recovered_false_ns[:, trial] = recovered_false_n
        if trial % 100 == 99:
            print("Finished trial %s" % (trial+1))

    return recovered_xs, recovered_false_ps, recovered_false_ns


def compare_truth_and_estimates(membership_matrix, xs_file, f, fpr=0, fnr=0, saveM=True):
    """
    Get ground truth from true_infection_vectors_file and attempt to recover the ground truth.
    :param membership_matrix: membership matrix used for recovery
    :param true_infection_vectors_file: filename for true infection vectors
    :param f: population infection rate
    :param fpr: false positive rate
    :param fnr: false negative rate
    :return:
    """
    xs = np.genfromtxt(xs_file, delimiter=',')
    num_samples, num_trials = xs.shape
    pool_results, fps, fns = simulate_pool_results(xs, membership_matrix, fpr, fnr)

    recovered_xs, recovered_fps, recovered_fns = recover_pool_results(membership_matrix,
                                                                      pool_results,
                                                                      fpr, fnr, f, test=True)

    check_optimality(xs, recovered_xs, fps, recovered_fps, fns, recovered_fns, fpr, fnr, f)

    accuracy = 1 - (xs != recovered_xs).sum(0) / num_samples
    num_fp = ((xs == 0) * (recovered_xs == 1)).sum(0)
    num_fn = ((xs == 1) * (recovered_xs == 0)).sum(0)

    info = {"accuracy": accuracy.tolist(),
            "num_fp": num_fp.tolist(),
            "num_fn": num_fn.tolist()}

    if saveM:
        info["membership_matrix"] = membership_matrix.tolist()

    return info


def check_optimality(xs, recovered_xs, fps, recovered_fps, fns, recovered_fns, fpr, fnr, f):
    """
    Check whether the ILP solver finds an optimal solution.
    :param xs: true infection vectors
    :param recovered_xs: recovered infection vectors
    :param verbose: set verbose to True to surpress print statements
    """
    fpr, fnr, f = check_inputs(fpr, fnr, f)

    def objective(xs, fps, fns):
        """We want to minimize this objective. """
        Wp = - np.log(fpr / (1 - fpr))  # Weight for false positives
        Wn = - np.log(fnr / (1 - fnr))  # Weight for false negatives
        Wx = - np.log(f / (1 - f))  # Weight for all positives
        return np.sum(xs) * Wx + np.sum(fps) * Wp + np.sum(fns) * Wn

    _, num_trials = xs.shape
    for trial in range(num_trials):
        x, recovered_x = xs[:, trial], recovered_xs[:, trial]
        num_errors = np.sum(x != recovered_x)
        objective_true = objective(xs, fps, fns)
        objective_recovered = objective(recovered_xs, recovered_fps, recovered_fns)
        if num_errors != 0 and objective_true < objective_recovered:
            with open('optimality.txt', 'w') as file:
                file.write("ILP solver fails to find the optimal the objective for trail %s of f = %s \n" % (trial, f))
                file.write("x = %s \n" % list(x))
                file.write("recovered_x = %s \n \n \n" % list(recovered_x))


def test_RS(f, num_trials, fpr=0, fnr=0):
    """
    this is the membership matrix by Shental et al.
    Download the file from https://github.com/NoamShental/PBEST/blob/master/mFiles/poolingMatrix.mat
    """
    matrix_file = scipy.io.loadmat('./data/poolingMatrix.mat')
    M = matrix_file['poolingMatrix']
    return test_M(M, f, 384, fpr, fnr, num_trials)


def test_M(M, f, n, fpr=0, fnr=0, num_trials=1000):
    xs_file = './data/n%s-f%.4f-numTrials%s.csv' % (n, f, num_trials)
    return compare_truth_and_estimates(M, xs_file, f, fpr, fnr, saveM=False)
