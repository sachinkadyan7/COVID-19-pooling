import math


def optimal_pool_size(f, fnr, fpr):
    """
    P(pool negative) = 0.5
    :param f: population infection rate
    :param fnr: false negative rate
    :param fpr: false positive rate
    :return: the optimal pool size.
    """
    return math.log((0.5-fnr)/(1-fpr - fnr), 1-f)


def optimal_column_weight(f, fnr, fpr, T, n):
    pool_size = optimal_pool_size(f, fnr, fpr)
    return round(pool_size * T / n)


def H(p):
    """
    Compute the entropy of X ~ Ber(p).
    :param p: Pr(X = 1)
    :return: entropy of X
    """
    return -p * math.log(p, 2) - (1 - p) * math.log(1 - p, 2)


def minT(f, n):
    """
    The minimum number of tests for n samples from infection rate f.
    :param f: infection rate
    :param n: number of samples
    :return: the lower bound for the number of tests, based on Shannon's theorem.
    """
    return math.ceil(n * H(f))


def entropy(pool_size, f):
    p = (1 - f) ** pool_size
    return H(p)
