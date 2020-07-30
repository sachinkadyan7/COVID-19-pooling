import numpy as np


def convert_labels_to_digits(labels, base, D):
    labels = np.asarray(labels)
    base_powers = base ** np.arange(D - 1, -1, -1)
    print("Powers: ", base_powers)
    digits = (labels.reshape(labels.shape + (1,)) // base_powers) % base
    print("\nlabels_in_digits\n", digits)
    return digits