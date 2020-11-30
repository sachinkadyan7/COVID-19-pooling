from mip import *
from util import check_inputs


def solve_mip(membership_matrix, pool_result, fpr, fnr, f, test=False):
    """
    :param membership_matrix: a numpy array, shape (num_pools * num_samples)
    :param pool_result: a numpy array with num_pools entries
    :param fpr: a float between 0 and 1 indicating false positive rate
    :param fnr: a float between 0 and 1 indicating false negative rate
    :param f: a float between 0 and 1 indicating population infection rate
    :return: the recovered x, false_positives, false_negative
    """

    # Check inputs
    fpr, fnr, f = check_inputs(fpr, fnr, f)

    num_pools, num_samples = membership_matrix.shape

    # Create model
    m = Model()

    # Add variables
    x = [m.add_var(name=str(i), var_type=BINARY) for i in range(num_samples)]
    pool_false_positives = [m.add_var(name=str(i + num_samples), var_type=BINARY) for i in range(num_pools)]
    pool_false_negatives = [m.add_var(name=str(i + num_samples+num_pools), var_type=BINARY) for i in range(num_pools)]

    b = [m.add_var(var_type=BINARY) for _ in range(num_pools)]

    # Add constraints
    for i in range(num_pools):
        if pool_result[i] == 0:
            m += xsum(x * membership_matrix[i, :]) - 0.5 + 0.5 * b[i] >= 0
            m += xsum(x * membership_matrix[i, :]) - 0.5 - num_pools * (1 - b[i]) <= 0
            m += pool_false_negatives[i] == 1 - b[i]
        elif pool_result[i] == 1:
            m += xsum(x * membership_matrix[i, :]) + 0.5 * pool_false_positives[i] - 0.5 >= 0
            m += xsum(x * membership_matrix[i, :]) - num_pools * (1 - pool_false_positives[i]) - 0.5 <= 0

    # Objective function
    Wp = - np.log(fpr/(1-fpr))      # Weight for false positives
    Wn = - np.log(fnr/(1-fnr))      # Weight for false negatives
    Wx = - np.log(f/(1-f))          # Weight for all positives

    m.objective = minimize(xsum(pool_false_positives[i] for i in range(num_pools))*Wp
                           + xsum(pool_false_negatives[i] for i in range(num_pools))*Wn
                           + xsum(x[i] for i in range(num_samples))*Wx)

    if test:
        status = m.optimize(max_seconds=60)
    else:
        status = m.optimize(max_seconds=600)

    recovered_x = np.zeros(num_samples, dtype=bool)
    recovered_false_p = np.zeros(num_pools, dtype=bool)
    recovered_false_n = np.zeros(num_pools, dtype=bool)

    if status == OptimizationStatus.OPTIMAL or status == OptimizationStatus.FEASIBLE:
        for i in range(num_samples):
            recovered_x[i] = m.vars[i].x
        for i in range(num_pools):
            recovered_false_p[i] = m.vars[i + num_samples].x
            recovered_false_n[i] = m.vars[i + num_samples + num_pools].x

    assert np.all(np.sign(membership_matrix @ recovered_x) + recovered_false_p - recovered_false_n == pool_result)

    return recovered_x, recovered_false_p, recovered_false_n
