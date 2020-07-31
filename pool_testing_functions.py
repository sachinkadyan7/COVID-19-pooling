import numpy as np
import itertools


"""
Creates a global membership matrix to denote membership of each sample in each pool.

    Membership matrix: 
        Rows in the matrix represent the individual samples
        The columns of a particular membership matrix (i.e. a particular poolset) denote the pools
        in that poolset.
    Say we have D=4, d=2. ijkl (D choose d) = ij, ik, il, jk, jl, kl
    So num(D choose d) = 6. Means we will have 6 poolsets.
    Each poolset will have a max base^d pools, and each pool would have max base^(D-d) samples.
    Putting all the metrics here:
    maxN = base ** D, i.e. max samples we can accommodate
    num_pools_in_poolset = base ** d
    max_num_ind_i_pool = base ** (D-d)
"""
def construct_membership_matrix(labels, D, d, base):
    print("\nConstructing membership matrix")

    # Number of samples available to us
    num_samples = labels.shape[0]
    # Max number of pools in a particular poolset
    num_pools_in_poolset = base ** d

    print("num_samples", num_samples)
    print("num_pools_in_poolset", num_pools_in_poolset)

    combinations_of_digit_places = itertools.combinations(range(D), d)
    print("combinations_of_digit_places", combinations_of_digit_places)

    base_powers = base ** np.arange(d-1, -1, -1)
    membership_matrix_global = None

    for psd in combinations_of_digit_places:
        print("\nPool selecting digits ", psd)

        selected_digits_in_labels = labels[:, psd]
        print("\nSelected digits\n", selected_digits_in_labels)
        pool_numbers = np.sum(selected_digits_in_labels * base_powers, axis=1)
        print("\npool numbers", pool_numbers)

        membership_matrix = np.zeros((num_samples, num_pools_in_poolset), dtype=int)
        membership_matrix[range(labels.shape[0]), pool_numbers] = 1
        print("\nMembership matrix\n", membership_matrix)

        if membership_matrix_global is None:
            membership_matrix_global = membership_matrix
        else:
            membership_matrix_global = np.hstack((membership_matrix_global, membership_matrix))

    print("\nmembership_matrix_global\n", membership_matrix_global)
    return membership_matrix_global


def perform_testing_of_pools(infection_samples, membership_matrix_global):
    result = np.dot(infection_samples, membership_matrix_global)
    print("result", result)
    return result
