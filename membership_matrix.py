import numpy as np


class MembershipMatrix:
    def __init__(self, num_samples):
        self.num_samples = num_samples
        self.num_pools = 0
        self.membership_matrix = None

    def multiply(self, vector):
        return np.dot(vector, self.membership_matrix)

    def add_poolset(self, poolset_membership_matrix):
        if self.membership_matrix is None:
            self.membership_matrix = poolset_membership_matrix
        else:
            self.membership_matrix = np.hstack((self.membership_matrix, poolset_membership_matrix))
        self.num_pools += poolset_membership_matrix.shape[1]

    def get_matrix(self):
        return self.membership_matrix

    def __str__(self):
        return self.membership_matrix.__str__()


def generate_const_row_weight_random_M(shape, m):
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
