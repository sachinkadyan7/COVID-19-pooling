from mip import *


def solve_mip(num_samples, membership_matrix, pool_result, num_pools):

    print("Solving")
    # Create model
    m = Model()

    # Add variables
    x = [ m.add_var(name=str(i), var_type=BINARY) for i in range(num_samples) ]
    false_pool = [ m.add_var(name=str(i+num_samples), var_type=BINARY) for i in range(num_pools) ]

    # Add constraints
    for i in range(num_pools):
        if pool_result[i] == 0:
            m += xsum(x * membership_matrix[:,i]) - false_pool[i] == 0
        elif pool_result[i] == 1:
            m += xsum(x * membership_matrix[:,i]) + false_pool[i] >= 1

    # Objective function
    m.objective = minimize(xsum(false_pool[i] for i in range(num_pools)))

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
                else:
                    print('false_pool{} : {}'.format(int(v.name)-num_samples, v.x))

    return solution, status
