import numpy as np
from mip import *


def solve_mip(membership_matrix, pool_result, fpr, fnr, f):
    """
    :param membership_matrix: a numpy array, shape (num_pools * num_samples)
    :param pool_result: a numpy array with num_pools entries
    :param fpr: a float between 0 and 1 indicating false positive rate
    :param fnr: a float between 0 and 1 indicating false negative rate
    :param f: a float between 0 and 1 indicating population infection rate
    :return:
    """

    # Check inputs
    assert f != 0, "Please input a non-zero infection rate."
    if fpr == 0:
        fpr = np.nextafter(0, 1)
    if fnr == 0:
        fnr = np.nextafter(0, 1)

    num_pools, num_samples = membership_matrix.shape

    # Create model
    m = Model()

    # Add variables
    x = [m.add_var(name=str(i), var_type=BINARY) for i in range(num_samples)]
    pool_false_positives = [m.add_var(name=str(i + num_samples), var_type=BINARY) for i in range(num_pools)]
    pool_false_negatives = [m.add_var(name=str(i + num_samples+num_pools), var_type=BINARY) for i in range(num_pools)]

    # Add constraints
    for i in range(num_pools):
        if pool_result[i] == 0:
            m += xsum(x * membership_matrix[i, :]) - pool_false_negatives[i] == 0
        elif pool_result[i] == 1:
            m += xsum(x * membership_matrix[i, :]) + pool_false_positives[i] >= 1

    # Objective function
    Wp = - np.log(fpr/(1-fpr))      # Weight for false positives
    Wn = - np.log(fnr/(1-fnr))      # Weight for false negatives
    Wx = - np.log(f/(1-f))          # Weight for all positives
    m.objective = minimize(xsum(pool_false_positives[i] for i in range(num_pools))*Wp
                           + xsum(pool_false_negatives[i] for i in range(num_pools))*Wn
                           + xsum(x[i] for i in range(num_samples))*Wx)

    status = m.optimize()

    recovered_x = np.zeros(num_samples, dtype=bool)
    recovered_false_p = np.zeros(num_pools, dtype=bool)
    recovered_false_n = np.zeros(num_pools, dtype=bool)

    if status == OptimizationStatus.OPTIMAL or status == OptimizationStatus.FEASIBLE:
        for v in m.vars:
            if abs(v.x) > 1e-6:
                if int(v.name) < num_samples:
                    recovered_x[int(v.name)] = 1
                elif num_samples <= int(v.name) < num_samples + num_pools:
                    recovered_false_p[int(v.name) - num_samples] = 1
                else:
                    recovered_false_n[int(v.name) - num_samples - num_pools] = 1

    return recovered_x, recovered_false_p, recovered_false_n, status
