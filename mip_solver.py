from mip import *


def solve_mip(num_samples, membership_matrix, pool_result, num_pools, fpr, fnr, f):

    print("Solving")
    # Create model
    m = Model()

    # Add variables
    x = [ m.add_var(name=str(i), var_type=BINARY) for i in range(num_samples) ]
    pool_false_positives = [ m.add_var(name=str(i+num_samples), var_type=BINARY) for i in range(num_pools) ]
    pool_false_negatives = [ m.add_var(name=str(i+num_samples+num_pools), var_type=BINARY) for i in range(num_pools) ]

    # Add constraints
    for i in range(num_pools):
        if pool_result[i] == 0:
            m += xsum(x * membership_matrix[:,i]) - pool_false_positives[i] == 0
        elif pool_result[i] == 1:
            m += xsum(x * membership_matrix[:,i]) + pool_false_negatives[i] >= 1

    # Objective function
    Wp = - np.log(fpr/(1-fpr))      # Weight for false positives
    Wn = - np.log(fnr/(1-fnr))      # Weight for false negatives
    Wx = - np.log(f/(1-f))          # Weight for all positives
    m.objective = minimize(xsum(pool_false_positives[i] for i in range(num_pools))*Wp
                           + xsum(pool_false_negatives[i] for i in range(num_pools))*Wn
                           + xsum(x[i] for i in range(num_pools))*Wx)

    status = m.optimize()
    if status == OptimizationStatus.OPTIMAL:
        print('optimal solution: ')
    elif status == OptimizationStatus.FEASIBLE:
        print('feasible solution')
    elif status == OptimizationStatus.NO_SOLUTION_FOUND:
        print('No solution found yet')
    elif status == OptimizationStatus.INFEASIBLE or status == OptimizationStatus.INT_INFEASIBLE:
        print('No feasible solution found', status)
    else:
        print('Infeasible, unbounded or error', status)

    solution = np.zeros(num_samples, dtype=bool)
    if status == OptimizationStatus.OPTIMAL or status == OptimizationStatus.FEASIBLE:
        for v in m.vars:
            if abs(v.x) > 1e-6:
                if int(v.name) < num_samples:
                    print('{} : {}'.format(v.name, v.x))
                    solution[int(v.name)] = True
                elif num_samples < int(v.name) < num_samples+num_pools:
                    print('false_positive_pool {} : {}'.format(int(v.name)-num_samples, v.x))
                else:
                    print('false_negative_pool {} : {}'.format(int(v.name)-num_samples-num_pools, v.x))

    return solution, status
