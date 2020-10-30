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


def H(p):
    """
    Compute the entropy of X ~ Ber(p).
    :param p: Pr(X = 1)
    :return: entropy of X
    """
    return -p * math.log(p) - (1 - p) * math.log(1 - p)
