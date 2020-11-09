import numpy as np
import random
import itertools
import json
import os

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


def simulate_x(num_samples, num_trials, f, filename):
    """
    Code to generate infection vector for testing.
    :param num_samples: integer, number of samples.
    :param num_trials: integer, number of trials.
    :param f: population infection rate.
    :param filename: filename, file can be found in ./tests/data/ folder.
    :return: None, saves the vectors to a csv file.
    """
    rows = num_samples  # 384
    cols = num_trials  # 100

    xs = np.empty(shape=(num_samples, num_trials))

    for r in range(rows):
        for c in range(cols):
            xs[r][c] = int(np.random.binomial(1, f, 1))  # binomial(n,p,trials)

    print("On average, %.2f positives in each trail." % np.average(xs.sum(0)))
    np.savetxt("./tests/data/" + filename, xs, delimiter=',')  # convert to csv
    return xs


def simulate_results(xs_filename, fnr, fpr):
    """
    Simulate pooling results.
    :param filename: filename for infection vector
    :param fnr: false negative rate
    :param fpr: false positive rate
    :return: None, saves the vectors to a csv file.
    """
    test_dir = os.getcwd() + 'tests/data/'
    xs = np.genfromtxt(test_dir + xs_filename, delimiter=',')
    Mxs =