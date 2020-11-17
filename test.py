import numpy as np
import json
import os
from mip_solver import solve_mip
from util import check_inputs, simulate_pool_results
import scipy.io


def recover_pool_results(membership_matrix, pool_results, fpr, fnr, f, print_every=10):
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
        if trial % print_every == 0:
            print("Starting trail %s ..." % trial)
        pool_result = pool_results[:, trial]
        recovered_x, recovered_false_p, recovered_false_n = solve_mip(membership_matrix, pool_result, fpr, fnr, f)
        recovered_xs[:, trial] = recovered_x
        recovered_false_ps[:, trial] = recovered_false_p
        recovered_false_ns[:, trial] = recovered_false_n

    return recovered_xs, recovered_false_ps, recovered_false_ns


def compare_truth_and_estimates(membership_matrix, true_infection_vectors_file, fpr, fnr, f):
    """
    Get ground truth from true_infection_vectors_file and attempt to recover the ground truth.
    :param membership_matrix: membership matrix used for recovery
    :param true_infection_vectors_file: filename for true infection vectors
    :param f: population infection rate
    :param fpr: false positive rate
    :param fnr: false negative rate
    :param verbose: set verbose to True to surpress print statement
    :return:
    """
    xs = np.genfromtxt(true_infection_vectors_file, delimiter=',')
    pool_results, fps, fns = simulate_pool_results(xs, membership_matrix, fpr, fnr)

    recovered_xs, recovered_fps, recovered_fns = recover_pool_results(membership_matrix,
                                                                      pool_results,
                                                                      fpr, fnr, f)

    check_optimality(xs, recovered_xs, fps, recovered_fps, fns, recovered_fns, fpr, fnr, f)

    num_errors = (xs != recovered_xs).sum()
    num_fp = ((xs == 0) * (recovered_xs == 1)).sum()
    num_fn = ((xs == 1) * (recovered_xs == 0)).sum()

    info = {"membership_matrix": membership_matrix.tolist(),
            "num_errors": int(num_errors),
            "num_fp": int(num_fp),
            "num_fn": int(num_fn)}

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
            print("ILP solver fails to find the optimal the objective for trail %s" % trial)


def test_random_M(m, k, n, T, fpr, fnr, num_random_matrices, COVID_dir, generate_matrix, print_every=5):
    """
    Saves the number of errors to ./test/results/
    :param m: constant row weight
    :param k: expected number of positives among 384 samples
    :param n: number of samples
    :param T: number of tests
    :param num_random_matrices: test num_trails random membership matrices
    :param COVID_dir: the path to 'COVID-19-pooling'
    :param generate_matrix: generate a random matrix using either 'generate_const_col_weight'
    or 'generate_doubly_regular_col'
    """
    folder_name = generate_matrix.__name__

    if not os.path.exists(COVID_dir + "/tests/results/"):
        os.makedirs(COVID_dir + "/tests/results/")
    if not os.path.exists(COVID_dir + "/tests/results/%s/" % folder_name):
        os.makedirs(COVID_dir + "/tests/results/%s/" % folder_name)

    f = k / n
    test_file = COVID_dir + '/tests/data/x-f-%s-384.csv' % k
    infos = []

    outfile_name = COVID_dir + "/tests/results/%s/m%s-k%s-n%s-T%s-numM%s.txt" % (folder_name, m, k, n, T, num_random_matrices)

    for i in range(num_random_matrices):
        if i % print_every == 0:
            print("Starting matrix %s" % i)
        matrix = generate_matrix((T, n), m)
        info = compare_truth_and_estimates(matrix, test_file, fpr, fnr, f)
        infos.append(info)

    with open(outfile_name, 'w') as outfile:
        json.dump(infos, outfile)


def get_num_errors(results_dir, n, k, T, num_random_matrices, weights, error_type):
    x = []
    y = []
    average_num_errors = []

    for m in weights:
        with open(results_dir + 'm%s-k%s-n%s-T%s-numM%s.txt' % (m, k, n, T, num_random_matrices)) as file:
            data = json.load(file)
            total_errors = 0
            for result in data:
                num_errors = result[error_type]
                total_errors += num_errors
                x.append(m)
                y.append(num_errors)
        average_num_errors.append(total_errors / num_random_matrices)

    return x, y, average_num_errors


def get_accuracy(COVID_dir, results_dir, n, k, T, num_trials, num_random_matrices, row_weights, error_type):
    xs = np.genfromtxt(COVID_dir + '/tests/data/x-f-%s-384.csv' % k, delimiter=',')
    total_positives = xs.sum()
    total_negatives = n * num_trials - total_positives

    denominator = {'num_errors': n * num_trials, 'num_fp': total_negatives, 'num_fn': total_positives}

    x = []
    y = []
    average_accuracy = []

    for m in row_weights:
        with open(results_dir + 'm%s-k%s-n%s-T%s-numM%s.txt' % (m, k, n, T, num_random_matrices)) as file:
            data = json.load(file)
            total_errors = 0
            for result in data:
                num_errors = result[error_type]
                total_errors += num_errors
                x.append(m)
                y.append(1 - num_errors / denominator[error_type])
        average_accuracy.append(1 - total_errors / (denominator[error_type] * num_random_matrices))

    return x, y, average_accuracy


def test_shental(shental_matrix_filepath, k, fpr, fnr):
    """
    this is the membership matrix by Shental et al.
    Download the file from https://github.com/NoamShental/PBEST/blob/master/mFiles/poolingMatrix.mat
    """
    matrix_file = scipy.io.loadmat(shental_matrix_filepath)
    membership_matrix = matrix_file['poolingMatrix']
    f = k/384
    file = os.getcwd() + '/data/x-f-%s-384.csv' % k
    return compare_truth_and_estimates(membership_matrix, file, fpr, fnr, f)

