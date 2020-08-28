"""
Sample usage:
python main.py -f 0.002 -N 4000 -B 8 -D 4 -d 2 -eps_fp 0.001 -eps_fn 0.001 --runs 100 --name test_report
"""

import argparse
import datetime
import os

import numpy as np
from mip import OptimizationStatus

import label_conversion
import util
from pool_testing_functions import perform_testing_of_pools, get_membership_matrix
from mip_solver import solve_mip

parser = argparse.ArgumentParser()
parser.add_argument('--runs',
                    type=int,
                    help='The number of runs in the simulation')

parser.add_argument('--name',
                    type=str,
                    help='The name of the report to save')

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

parser.add_argument('-eps_fp',
                    type=float,
                    help='False positive error rate',
                    default=0.0)

parser.add_argument('-eps_fn',
                    type=float,
                    help='False negative error rate',
                    default=0.0)

args = parser.parse_args()


def main():

    # np.random.seed(42)

    params = {'f': args.f,
              'N': args.N,
              'B': args.B,
              'D': args.D,
              'd': args.d,
              'eps_fp': args.eps_fp,
              'eps_fn': args.eps_fn,
              'name': args.name,
              'runs': args.runs}

    start(params)


def start(params):
    report_name = datetime.datetime.now().__str__() if params["name"] is None else params["name"]
    directory = os.path.join(os.getcwd(), 'reports', report_name)
    os.mkdir(directory)
    print("Saving run to directory ", directory)
    util.write_json(params, os.path.join(directory, 'params.json'))
    simulation_statistics = {'tp': 0,
                             'tn': 0,
                             'fp': 0,
                             'fn': 0,
                             'infeasible_times': 0}
    print("Metrics")
    util.calc_metrics(params["D"], params["d"], params["B"], params["N"])
    for run in range(params["runs"]):
        print("\n\nSimulation run ", run)

        infection_samples = util.draw_samples(params["N"], params["f"])
        print("Infection samples vector", sum(infection_samples))
        print("Indices of positive samples", [i for i, x in enumerate(infection_samples) if x])

        labels = util.assign_labels(params["N"], params["B"], params["D"])
        print("Labels", len(labels), labels)

        labels_in_digits = label_conversion.convert_labels_to_digits(labels, params["B"], params["D"])
        membership_matrix_global, psd_global = get_membership_matrix(labels_in_digits, params["D"], params["d"], params["B"])
        result = perform_testing_of_pools(infection_samples, membership_matrix_global, params["eps_fp"], params["eps_fn"])

        # Recover individual samples
        solution, status = solve_mip(params["N"], membership_matrix_global.get_matrix(), result, result.shape[0],
                                     params["eps_fp"], params["eps_fn"], params["f"])

        comparison_array = np.column_stack((infection_samples, solution, labels_in_digits))
        np.savetxt(os.path.join(directory, str(run)), comparison_array, fmt='%.18e %.18e %d %d %d %d')

        simulation_statistics['tp'] += np.count_nonzero(np.logical_and(infection_samples == True, solution == True))
        simulation_statistics['tn'] += np.count_nonzero(np.logical_and(infection_samples == False, solution == False))
        simulation_statistics['fp'] += np.count_nonzero(np.logical_and(infection_samples == False, solution == True))
        simulation_statistics['fn'] += np.count_nonzero(np.logical_and(infection_samples == True, solution == False))
        if status != OptimizationStatus.OPTIMAL and status != OptimizationStatus.FEASIBLE:
            simulation_statistics['infeasible_times'] += 1
    # Print results
    simulation_statistics = util.calculate_stats(simulation_statistics)
    print("Stats: ", simulation_statistics)
    util.write_json(simulation_statistics, os.path.join(directory, 'stats.json'))


if __name__ == '__main__':
    main()
