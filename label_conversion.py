import numpy as np


def convert_labels_to_digits(labels, base, D):
    labels = np.asarray(labels)
    base_powers = base ** np.arange(D - 1, -1, -1)
    print("Powers: ", base_powers)
    digits = (labels.reshape(labels.shape + (1,)) // base_powers) % base
    return digits


def convert_labels_to_digits_old(labels, base, D):
    labels_in_digits = [None] * len(labels)
    for i, label in enumerate(labels):
        labels_in_digits[i] = convert_label_to_digits_old(label, base, D)
    return labels_in_digits


def reVal(num):
    if 0 <= num <= 9:
        return chr(num + ord('0'));
    else:
        return chr(num - 10 + ord('A'));


def convert_label_to_digits_old(label_num, base, D):
    index = 0  # Initialize index of result
    res = []

    # Convert input number is given base
    # by repeatedly dividing it by base
    # and taking remainder
    while label_num > 0:
        res += reVal(label_num % base)
        label_num = int(label_num / base)

    while len(res) < D:
        res += '0'

        # Reverse the result
    res = res[::-1]

    return res
