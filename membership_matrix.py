import numpy as np


def generate_const_row_weight(shape, m):
    """
    :param shape: shape of the membership matrix, (num_pools, num_samples).
    :param m: row weight of the membership matrix.
    :return: a randomly generated matrix with row weights m.
    """
    M = np.zeros(shape)
    T, n = M.shape
    for i in range(T):
        indices = np.random.choice(n, m, replace=False)
        for index in indices:
            M[i, index] = 1
    assert np.all(M.sum(1) == m)

    return M


def generate_const_col_weight(shape, m):
    """
    :param shape: shape of the membership matrix, (num_pools, num_samples).
    :param m: column weight of the membership matrix.
    :return: a randomly generated matrix with column weights m.
    """
    return generate_const_row_weight(shape[::-1], m).T


def generate_doubly_regular_row(shape, m):
    """
    :param shape: shape of the membership matrix, (num_pools, num_samples).
    :param m: row weight of the membership matrix.
    :return: a randomly generated matrix with row weight m and nearly constant column weight.
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

    column_weights = M.sum(0)
    if goal > M.sum() / shape[1]:
        assert np.logical_or(column_weights == goal, column_weights == goal - 1).all()
    elif goal < M.sum() / shape[1]:
        assert np.logical_or(column_weights == goal, column_weights == goal + 1).all()
    else:
        assert np.all(column_weights == goal)
    return M


def generate_doubly_regular_col(shape, m):
    """
    :param shape: shape of the membership matrix, (num_pools, num_samples).
    :param m: column weight of the membership matrix.
    :return: a randomly generated doubly regular matrix with column weight m (and nearly constant row weight).
    """
    M = generate_doubly_regular_row(shape[::-1], m).T
    return M
