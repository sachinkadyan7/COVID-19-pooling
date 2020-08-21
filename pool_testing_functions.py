import numpy as np
import itertools

from membership_matrix import MembershipMatrix

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
def get_membership_matrix(labels, D, d, base):
    print("\nConstructing membership matrix")

    # Number of samples available to us
    num_samples = labels.shape[0]
    # Max number of pools in a particular poolset
    num_pools_in_poolset = base ** d

    print("num_samples", num_samples)
    print("num_pools_in_poolset", num_pools_in_poolset)

    combinations_of_digit_places = itertools.combinations(range(D), d)
    print("combinations_of_digit_places", combinations_of_digit_places)

    membership_matrix_global, pool_selecting_digits_global = \
        construct_membership_matrix(base,
                                    combinations_of_digit_places,
                                    d, labels,
                                    num_pools_in_poolset,
                                    num_samples)

    print("\nmembership_matrix_global\n", membership_matrix_global)
    return membership_matrix_global, pool_selecting_digits_global


def construct_membership_matrix(base, combinations_of_digit_places, d, labels, num_pools_in_poolset, num_samples):
    base_powers = base ** np.arange(d - 1, -1, -1)
    membership_matrix_global = MembershipMatrix(num_samples)
    pool_selecting_digits_global = []
    for psd in combinations_of_digit_places:
        print("\nPool selecting digits ", psd)

        selected_digits_in_labels = labels[:, psd]
        print("\nSelected digits\n", selected_digits_in_labels)
        pool_numbers = np.sum(selected_digits_in_labels * base_powers, axis=1)
        print("\npool numbers", pool_numbers)

        membership_matrix = np.zeros((num_samples, num_pools_in_poolset), dtype=int)
        membership_matrix[range(labels.shape[0]), pool_numbers] = 1
        print("\nMembership matrix\n", membership_matrix)

        membership_matrix_global.add_poolset(membership_matrix)
        pool_selecting_digits_global.append(psd)
    return membership_matrix_global, pool_selecting_digits_global


def perform_testing_of_pools(infection_samples, membership_matrix_global, eps_fp, eps_fn):
    result = membership_matrix_global.multiply(infection_samples)
    thresholded_result = np.clip(result, 0, 1)
    print("result", thresholded_result)

    num_pools = thresholded_result.shape[0]
    inverse_thresholded_result = np.logical_not(thresholded_result)
    pools_plus_errors = thresholded_result

    # False positives can happen with an error probability eps_fp
    if eps_fp != 0.0:
        pools_plus_errors = np.where(np.logical_and(thresholded_result == 0, np.random.random(num_pools) <= eps_fp),
                                     inverse_thresholded_result, thresholded_result)
        print("eps_fp", pools_plus_errors)
    # False negatives can happen with an error probability eps_fn
    if eps_fn != 0.0:
        pools_plus_errors = np.where(np.logical_and(thresholded_result == 1, np.random.random(num_pools) <= eps_fn),
                                     inverse_thresholded_result, pools_plus_errors)
        print("eps_fn", pools_plus_errors)

    print("resEps", pools_plus_errors)
    return pools_plus_errors
