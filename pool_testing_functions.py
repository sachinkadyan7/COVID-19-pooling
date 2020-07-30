import numpy as np
import itertools


"""Perform testing of pools based on individual samples"""
def perform_testing_of_pools_old(infection_samples, labels, d):
    pools = {}

    for i, label in enumerate(labels):
        print("label ", i, label)
        common_digits = "".join(label[:d])
        print("common digits ", i, common_digits)
        if common_digits in pools:
            pools[common_digits] = pools[common_digits] or infection_samples[i]
        else:
            pools[common_digits] = infection_samples[i]

    # Insert True or False i.e. false positive or false negative
    # at a rate of error eps.

    return pools


def optimized_testing_of_pools_old(infection_samples, labels, D, d, base):
    # pools = np.array((base, d))
    # pools[tuple(labels[:, :d])] = np.logical_and(pools[tuple(labels[:, :d])])

    # Membership matrix: rows in the matrix represent the individual samples
    # The columns of a particular membership matrix (i.e. a particular poolset) denote the pools
    # in that poolset.
    # Say we have D=4, d=2. ijkl (D choose d) = ij, ik, il, jk, jl, kl
    # So num(D choose d) = 6. Means we will have 6 poolsets.
    # Each poolset will have a max base**d pools, and each pool would have max base**(D-d) samples.
    # Putting all the metrics here:
    # maxN = base ** D, i.e. max samples we can accommodate
    # num_pools_in_poolset = base ** d
    # max_num_ind_i_pool = base ** (D-d)

    num_samples = labels.shape[0]

    # Maximum samples we can handle
    maxN = base ** D
    print("maxN", maxN)

    # Number of pools in a particular poolset
    num_pools_in_poolset = base ** d
    print("num_pools_in_poolset", num_pools_in_poolset)

    # We choose d number of positions from available number of positions D in label.
    digit_places_in_label = range(D)
    combinations_of_digit_places = itertools.combinations(digit_places_in_label, d)
    Membership_1st_Poolset = np.zeros((labels.shape[0], num_pools_in_poolset), dtype=np.int)
    psd = list(combinations_of_digit_places)
    psd0 = psd[0]
    print("PSD ", psd)
    print("PSD1 ", psd0)

    pool = np.empty((labels.shape[0], d), dtype=np.int)
    print("\npool.shape", pool.shape)

    # Assign membership in first poolset for all samples
    # For each label, take the digits in consideration and mark those as true.
    print("\nlables\n", type(labels))
    for i in range(labels.shape[0]):
        print(i, labels[i, psd0])
        pool[i] = labels[i, psd0]
        Membership_1st_Poolset[i,pool[i]] = 1
    print("\npool", pool)
    print("\nMembership poolset", Membership_1st_Poolset)

    pool_new = labels[:, psd0]
    base_powers = base ** np.arange(d-1, -1, -1)
    pool_numbers = np.sum(pool_new * base_powers, axis=1)
    print("\npool numbers", pool_numbers)
    Membership_1st_Poolset_new = np.zeros_like(Membership_1st_Poolset)
    Membership_1st_Poolset_new[range(labels.shape[0]), pool_numbers] = 1
    print("\npool_new", pool_new)
    print("\nMembership poolset new", Membership_1st_Poolset_new)


def optimized_testing_of_pools(infection_samples, labels, D, d, base):
    print("\noptimized_testing_of_pools_new")

    # Number of samples available to us
    num_samples = labels.shape[0]
    # Max number of pools in a particular poolset
    num_pools_in_poolset = base ** d

    print("num_samples", num_samples)
    print("num_pools_in_dataset", num_pools_in_poolset)

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
