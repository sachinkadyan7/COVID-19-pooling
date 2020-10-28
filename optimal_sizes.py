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
