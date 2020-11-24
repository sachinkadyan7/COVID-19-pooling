import numpy as np
import random
import itertools
import json, os

'''Create N population samples with infection rate f'''
def draw_samples(N, f):
    infection_samples = np.random.random(N) < f
    return infection_samples


'''Assign the N infection samples labels in the range 0 .. B^D-1'''
def assign_labels(N, B, D):
    # Randomly select N number of samples from the range of labels 0 .. B^D-1
    labels = random.sample(range(B ** D), N)
    return labels


'''Calculate the various metrics associated with the infection samples pooling'''
def calc_metrics(D, d, B, N):
    pool_labels = list(itertools.combinations(range(D), d))
    print("Pool labels ", len(pool_labels), [i for i in pool_labels])

    # Calculate L = number_of_pools
    number_of_pools = len(pool_labels) * B**d
    print("L = Number of pools", number_of_pools)

    # Calculate S = size_of_each_pool
    size_of_each_pool = N / B ** d
    print("S = Size of each pool", size_of_each_pool)

    # Calculate R = redundancy
    redundancy = len(pool_labels)
    print("R = Redundancy", redundancy)


def calculate_stats(stats):
    stats['accuracy'] = (stats['tp'] + stats['tn']) / (stats['tp']+stats['tn']+stats['fp']+stats['fn'])
    return stats


def write_json(dict_contents, filename):
    json_dump = json.dumps(dict_contents)
    f = open(filename, "w")
    f.write(json_dump)
    f.close()


def check_inputs(fpr, fnr, f):
    assert f != 0, "Please input a non-zero infection rate."
    if fpr == 0:
        fpr = np.nextafter(0, 1)
    if fnr == 0:
        fnr = np.nextafter(0, 1)
    return fpr, fnr, f


def simulate_x(n, f, filepath=None, num_trials=100):
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
    return xs


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

