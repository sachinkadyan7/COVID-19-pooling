import numpy as np
import json
from mip_solver import solve_mip
from util import check_inputs, simulate_pool_results
from membership_matrix import generate_doubly_regular_col
import scipy.io
import os


def recover_pool_results(membership_matrix, pool_results, fpr, fnr, f):
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
        recovered_x, recovered_false_p, recovered_false_n = solve_mip(membership_matrix, pool_result, fpr, fnr, f)
        recovered_xs[:, trial] = recovered_x
        recovered_false_ps[:, trial] = recovered_false_p
        recovered_false_ns[:, trial] = recovered_false_n

    return recovered_xs, recovered_false_ps, recovered_false_ns


def compare_truth_and_estimates(membership_matrix, xs_file, fpr, fnr, f):
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
    xs = np.genfromtxt(xs_file, delimiter=',')
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
            with open('optimality.txt', 'w') as file:
                file.write("ILP solver fails to find the optimal the objective for trail %s of f = %s \n" % (trial, f))
                file.write("x = %s \n" % list(x))
                file.write("recovered_x = %s \n \n \n" % list(recovered_x))


def test_M(m, n, T, f, fpr, fnr, num_trials=100, test_file=None, generate_matrix=generate_doubly_regular_col, num_M=25, print_every=1):
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

    if not os.path.exists("./results/"):
        os.makedirs("./results/")
    if not os.path.exists("./results/%s/" % folder_name):
        os.makedirs("./results/%s/" % folder_name)

    infos = []
    if test_file is None: 
      test_file = "./data/n%s-f%.4f-numTrials%s.csv" % (n, f, num_trials)
    
    outfile_name = "./results/%s/m%s-f%.4f-n%s-T%s-numM%s-numTrials%s.txt" % (folder_name, m, f, n, T,
                                                                            num_M, num_trials)

    for i in range(num_M):
        if i % print_every == 0:
            print("Starting matrix %s/%s" % (i, num_M))
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


def get_accuracy_m(n, f, T, weights, error_type="num_errors", num_trials=100, num_M=25, xs_file=None, results_folder='./results/generate_doubly_regular_col/'):
    if xs_file is None:
        xs_file = './data/n%s-f%.4f-numTrials%s.csv' % (n, f, num_trials)
    xs = np.genfromtxt(xs_file, delimiter=',')

    total_positives = xs.sum()
    total_negatives = n * num_trials - total_positives

    denominator = {'num_errors': n * num_trials, 'num_fp': total_negatives, 'num_fn': total_positives}

    x = []
    y = []
    average_accuracy = []

    for m in weights:
        with open(results_folder + 'm%s-f%.4f-n%s-T%s-numM%s-numTrials%s.txt' % (m, f, n, T, num_M, num_trials)) as file:
            data = json.load(file)
            total_errors = 0
            for result in data:
                num_errors = result[error_type]
                total_errors += num_errors
                x.append(m)
                y.append(1 - num_errors / denominator[error_type])
        average_accuracy.append(1 - total_errors / (denominator[error_type] * num_M))

    return x, y, average_accuracy


def get_accuracy_T(n, f, m, Ts, error_type="num_errors", num_trials=100, num_M=25, xs_file=None, results_folder='./results/generate_doubly_regular_col/'):
    if xs_file is None:
        xs_file = './data/n%s-f%.4f-numTrials%s.csv' % (n, f, num_trials)
    xs = np.genfromtxt(xs_file, delimiter=',')

    total_positives = xs.sum()
    total_negatives = n * num_trials - total_positives

    denominator = {'num_errors': n * num_trials, 'num_fp': total_negatives, 'num_fn': total_positives}

    x = []
    y = []
    average_accuracy = []

    for T in Ts:
        with open(results_folder + 'm%s-f%.4f-n%s-T%s-numM%s-numTrials%s.txt' % (m, f, n, T, num_M, num_trials))  as file:
            data = json.load(file)
            total_errors = 0
            for result in data:
                num_errors = result[error_type]
                total_errors += num_errors
                x.append(T)
                y.append(1 - num_errors / denominator[error_type])
        average_accuracy.append(1 - total_errors / (denominator[error_type] * num_M))

    return x, y, average_accuracy


def test_RS(f, fpr=0, fnr=0, num_trials=100):
    """
    this is the membership matrix by Shental et al.
    Download the file from https://github.com/NoamShental/PBEST/blob/master/mFiles/poolingMatrix.mat
    """
    matrix_file = scipy.io.loadmat('./data/poolingMatrix.mat')
    membership_matrix = matrix_file['poolingMatrix']
    file = './data/n384-f%.4f-numTrials%s.csv' % (f, num_trials)
    return compare_truth_and_estimates(membership_matrix, file, fpr, fnr, f)
