import numpy as np
import json
import os
from mip_solver import solve_mip

fpr = 0  # TODO: add errors
fnr = 0  # TODO: add errors

PRINT_EVERY = 10


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
        if trial % PRINT_EVERY == 0 and not verbose:
            print("Starting trail %s ..." % trial)
        pool_result = pool_results[:, trial]
        recovered_x, recovered_false_p, recovered_false_n = solve_mip(membership_matrix, pool_result, fpr, fnr, f)
        recovered_xs[:, trial] = recovered_x
        recovered_false_ps[:, trial] = recovered_false_p
        recovered_false_ns[:, trial] = recovered_false_n

    return recovered_xs, recovered_false_ps, recovered_false_ns


def compare_truth_and_estimates(membership_matrix, true_infection_vectors_file, f, fpr, fnr, verbose=False):
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

    pool_results = np.sign(membership_matrix @ xs)
    recovered_xs, recovered_false_ps, recovered_false_ns = recover_pool_results(membership_matrix,
                                                                                pool_results,
                                                                                fpr, fnr, f, verbose)

    num_errors = (xs != recovered_xs).sum()
    num_fp = ((xs == 0) * (recovered_xs == 1)).sum()
    num_fn = ((xs == 1) * (recovered_xs == 0)).sum()

    result = {"num_errors": int(num_errors), "num_fp": int(num_fp), "num_fn": int(num_fn)}

    if not verbose:
        accuracy = (xs == recovered_xs).sum() / xs.size
        print("=========================")
        print("%s errors: %s false positive(s), %s false negative(s)" % (num_errors, num_fp, num_fn))
        print("accuracy: %.2f %%" % (accuracy * 100))

    return xs, recovered_xs, recovered_false_ps, recovered_false_ns, result


def check_optimality(xs, recovered_xs, verbose=False):
    """
    TODO: This is only for noiseless data. For noisy measurement experiments, need to include  ||f|| and ||n|| in the
    objective."
    Check whether the ILP solver finds an optimal solution.
    :param xs: true infection vectors
    :param recovered_xs: recovered infection vectors
    :param verbose: set verbose to True to surpress print statements
    :return:
    """
    _, num_trials = xs.shape
    for trial in range(num_trials):
        x, recovered_x = xs[:, trial], recovered_xs[:, trial]
        num_errors = (x != recovered_x).sum()
        if num_errors != 0 and not verbose:
            print("||x|| = %s >= ||recovered_x||? %s" % (sum(x), sum(x) >= sum(recovered_x)))
        elif num_errors != 0 and sum(x) < sum(recovered_x):
            print("ILP solver fails to find the optimize the objective for trail %s" % trial)


def test_random_M(m, k, n, T, num_random_matrices, COVID_dir, generate_matrix, print_every=5, verbose=False):
    """
    Saves the number of errors to ./test/results/
    :param m: constant row weight
    :param k: expected number of positives among 384 samples
    :param n: number of samples
    :param T: number of tests
    :param num_random_matrices: test num_trails random membership matrices
    :param COVID_dir: the path to 'COVID-19-pooling'
    :param generate_matrix: generate a random matrix using either 'generate_const_row_weight'
    or 'generate_doubly_regular'
    """
    folder_names = {'generate_const_row_weight': 'const-row-weight', 'generate_doubly_regular': 'doubly-regular'}
    folder_name = folder_names[generate_matrix.__name__]

    if not os.path.exists(COVID_dir + "/tests/results/"):
        os.makedirs(COVID_dir + "/tests/results/")
    if not os.path.exists(COVID_dir + "/tests/results/%s/" % folder_name):
        os.makedirs(COVID_dir + "/tests/results/%s/" % folder_name)

    f = k / n
    test_file = COVID_dir + '/tests/data/x-f-%s-384.csv' % k
    results = []

    outfile_name = COVID_dir + "/tests/results/%s/m%s-k%s-n%s-T%s-numTrials%s.txt" % (folder_name, m, k, n, T, num_random_matrices)

    for i in range(num_random_matrices):
        if i % print_every == 0:
            print("Starting matrix %s" % i)
        matrix = generate_matrix((T, n), m)
        xs, recovered_xs, recovered_false_ps, recovered_false_ns, result = compare_truth_and_estimates(matrix, test_file, f, fpr, fnr, verbose=True)
        results.append(result)
        check_optimality(xs, recovered_xs, verbose=True)

    with open(outfile_name, 'w') as outfile:
        json.dump(results, outfile)

    if not verbose:
        num_errors = []
        for result in results:
            num_errors.append(result['num_errors'])
        average_errors = np.average(num_errors)
        print("======================")
        print("Test result for constant row weight = %s, infection rate %s/%s:" % (m, k, n))
        print("(based on %s membership matrices)" % num_random_matrices)
        print("Average Accuracy: %.2f " % (1 - average_errors / (n * 100)))
        print("======================")
