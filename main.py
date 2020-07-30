"""
Sample usage:
python main.py -f 0.01 -N 10000 -B 10 -D 4 -d 2 --runs 2
"""

import argparse
import numpy as np

import label_conversion
import util
from pool_testing_functions import perform_testing_of_pools, optimized_testing_of_pools

parser = argparse.ArgumentParser()
parser.add_argument('--runs',
                    type=int,
                    help='The number of runs in the simulation')

parser.add_argument('-f',
                    type=float,
                    help='Fraction of infected individuals')

parser.add_argument('-N',
                    type=int,
                    help='Number of individuals to consider')

parser.add_argument('-B',
                    type=int,
                    help='Base for the identifiers that we will allocate')

parser.add_argument('-D',
                    type=int,
                    help='Dimension (number of digits in base D')

parser.add_argument('-d',
                    type=int,
                    help='The number of digits that we fix for each pool')

parser.add_argument('-eps',
                    type=float,
                    help='Error rate',
                    default=0.0)

args = parser.parse_args()


def main():

    np.random.seed(42)

    print("Metrics")
    util.calc_metrics(args.D, args.d, args.B, args.N)

    for run in range(args.runs):
        print("\n\nSimulation run ", run)

        infection_samples = util.draw_samples(args.N, args.f)
        print("Infection samples vector", sum(infection_samples))
        print("Indices of positive samples", [i for i, x in enumerate(infection_samples) if x])

        labels = util.assign_labels(args.N, args.B, args.D)
        print("Labels", len(labels), labels)

        print("typeof", type(labels))

        perform_testing_of_pools(infection_samples, labels)

        # Recover individual samples

        # Print results


if __name__ == '__main__':
    # labels_in_digits = label_conversion.convert_labels_to_digits([6, 5, 1], 2, 3)
    # print(labels_in_digits)

    labels_in_digits = label_conversion.convert_labels_to_digits([6, 5, 1], 2, 3)
    print("lables in digits\n", labels_in_digits)

    infections = [True, False, True, False, True, False, True, False]
    membership_matrix_d2 = optimized_testing_of_pools(infections, labels_in_digits, 3, 1, 2)
    print("\nd=1", membership_matrix_d2)
    membership_matrix_d3 = optimized_testing_of_pools(infections, labels_in_digits, 3, 2, 2)
    print("\nd=2", membership_matrix_d3)

    result = perform_testing_of_pools(infections[:3], membership_matrix_d3)
