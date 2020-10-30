import numpy as np
from mip_solver import solve_mip
from membership_matrix import generate_const_row_weight_random_M


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
    result = {"xs": xs}

    pool_results = np.sign(membership_matrix @ xs)
    recovered_xs, recovered_false_ps, recovered_false_ns = recover_pool_results(membership_matrix,
                                                                                pool_results,
                                                                                fpr, fnr, f, verbose)

    result["recovered_xs"] = recovered_xs
    result["recovered_false_ps"] = recovered_false_ps
    result["recovered_false_ns"] = recovered_false_ns

    if not verbose:
        print("=========================")

    num_errors = (xs != recovered_xs).sum()
    num_fp = ((xs == 0) * (recovered_xs == 1)).sum()
    num_fn = ((xs == 1) * (recovered_xs == 0)).sum()
    accuracy = (xs == recovered_xs).sum() / xs.size

    result["num_errors"] = (xs != recovered_xs).sum()

    if not verbose:
        print("%s errors: %s false positive(s), %s false negative(s)" % (num_errors, num_fp, num_fn))
        print("accuracy: %.2f %%" % (accuracy * 100))

    return result


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
    for trial in range(100):
        x, recovered_x = xs[:, trial], recovered_xs[:, trial]
        num_errors = (x != recovered_x).sum()
        if num_errors != 0 and not verbose:
            print("||x|| = %s >= ||recovered_x||? %s" % (sum(x), sum(x) >= sum(recovered_x)))
        elif num_errors != 0 and sum(x) < sum(recovered_x):
            print("ILP solver fails to find the optimize the objective for trail %s" % trial)


def test_random_M(m, k, n, T, num_trails, COVID_dir, print_every=5):
    """
    Saves the number of errors to ./test/results/
    m: constant row weight
    k: expected number of positives among 384 samples
    n: number of samples
    T: number of tests
    num_trails: test num_trails random membership matrices
    """
    fpr, fnr, f = 0, 0, k / 384
    file = COVID_dir + '/tests/data/x-f-%s-384.csv' % k
    num_errors = []
    for i in range(num_trails):
        if i % print_every == 0:
            print("Starting trail %s" % i)
        matrix = generate_const_row_weight_random_M((T, n), m)
        result = compare_truth_and_estimates(matrix, file, f, fpr, fnr, verbose=True)
        num_errors.append(result['num_errors'])
        check_optimality(result['xs'], result['recovered_xs'], verbose=True)
    average_errors = np.average(num_errors)

    print("======================")
    print("Below is the test result for constant row weight = %s, infection rate %s/384" % (m, k))
    print("The result is based on %s simulations." % num_trails)
    print("Average number of total errors: %s" % average_errors)
    print("Average Accuracy: %.2f %%" % (average_errors / (n * T) * 100))
    print("======================")

    np.savetxt(COVID_dir + "/tests/results/m%s-k%s-n%s-T%s.csv" % (m, k, n, T), num_errors, delimiter=',')
