import numpy as np


def generate_const_row_weight(shape, m):
    """
    :param shape: shape of the membership matrix, (num_pools, num_samples)
    :param m: row weight of the membership matrix
    :return: a randomly generated matrix with row weight m.
    """
    random_membership_matrix = np.zeros(shape)
    num_pools, num_samples = random_membership_matrix.shape
    for i in range(num_pools):
        indices = np.random.choice(num_samples, m, replace=False)
        for index in indices:
            random_membership_matrix[i, index] = 1
    return random_membership_matrix


def generate_const_col_weight(shape, m):
    """
    :param shape: shape of the membership matrix, (num_pools, num_samples)
    :param m: column weight of the membership matrix
    :return: a randomly generated matrix with column weight m.
    """
    return generate_const_row_weight(shape[::-1], m).T


def generate_doubly_regular_row(shape, m):
    """
    :param shape: shape of the membership matrix, (num_pools, num_samples)
    :param m: row weight of the membership matrix
    :return: a randomly generated doubly regular matrix with row weight m (and nearly constant column weight).
    """
    M = generate_const_row_weight(shape, m)
    goal = round(M.sum() / shape[1])

    assert goal >= 1, "Please input a row weight at least num_samples / num_pools."

    indices = np.argsort(M.sum(0)).tolist()  # this is the list of increasing indices

    stop = False

    while not stop:
        missing_index = indices[0]
        excess_index = indices[-1]

        missing_column = M[:, missing_index]
        excess_column = M[:, excess_index]

        diff = set(np.where(excess_column)[0]) - set(np.where(missing_column)[0])

        swap_row_index = diff.pop()

        M[swap_row_index, excess_index] = 0
        M[swap_row_index, missing_index] = 1

        indices = np.argsort(M.sum(0)).tolist()
        stop = (M.sum(0)[indices[0]] >= goal - 1 and M.sum(0)[indices[-1]] == goal) or \
               (M.sum(0)[indices[0]] == goal and M.sum(0)[indices[-1]] <= goal + 1)

    return M


def generate_doubly_regular_col(shape, m):
    """
    :param shape: shape of the membership matrix, (num_pools, num_samples)
    :param m: column weight of the membership matrix
    :return: a randomly generated doubly regular matrix with column weight m (and nearly constant row weight).
    """
    return generate_doubly_regular_row(shape[::-1], m).T
