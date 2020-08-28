import matplotlib.pyplot as plt
import os
import json

import numpy as np

fx_values = [0.001, 0.002, 0.003, 0.005, 0.008, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1]
metric_values_tp = []
metric_values_fn = []

for f in fx_values:
    report_name = 'performance_hamming_fpis' + str(f) + '_fnis0.001'
    directory = os.path.join(os.getcwd(), 'reports', report_name)
    with open(os.path.join(directory, 'stats.json'), 'r') as file:
        stats = json.load(file)
        metric_values_tp.append(stats['tp'])
        metric_values_fn.append(stats['fn'])

plt.figure("TP and FN vs fp_error for hamming optimization")
plt.title('True positives and false negatives vs fp error')
plt.xscale('log')
plt.xlabel('false positive rate f')
plt.ylabel('Absolute values')
plt.tick_params(axis='both', which='both')
plt.grid(True, which='both', )
plt.plot(fx_values, metric_values_tp, label='TruePos')
plt.plot(fx_values, metric_values_fn, label='FalseNeg')
plt.legend()

metric_values_tp = []
metric_values_fn = []

for f in fx_values:
    report_name = 'performance_hamming_fpis0.001_fnis' + str(f)
    directory = os.path.join(os.getcwd(), 'reports', report_name)
    with open(os.path.join(directory, 'stats.json'), 'r') as file:
        stats = json.load(file)
        metric_values_tp.append(stats['tp'])
        metric_values_fn.append(stats['fn'])

plt.figure("TP and FN vs fn_error for hamming optimization")
plt.title('True positives and false negatives vs fn error')
plt.xscale('log')
plt.xlabel('false negative rate f')
plt.ylabel('Absolute values')
plt.tick_params(axis='both', which='both')
plt.grid(True, which='both', )
plt.plot(fx_values, metric_values_tp, label='TruePos')
plt.plot(fx_values, metric_values_fn, label='FalseNeg')
plt.legend()

plt.show()

# Histogram for distribution of number of false positives in each of 100 runs.
report_name = 'hamming_fpis0.001_fnis0.001'
directory = os.path.join(os.getcwd(), 'reports', report_name)
false_positive_counts = []
for i in range(100):
    with open(os.path.join(directory, str(i)), 'r') as file:
        comparison_array = np.loadtxt(file)
        false_positive_counts.append(np.count_nonzero(np.logical_and(np.logical_not(comparison_array[:,0]), comparison_array[:,1])))
        print(i, false_positive_counts[i])
plt.figure("Distribution of false positives in each of 100 runs")
plt.title('Number of false positives in each run')
plt.hist(false_positive_counts)
plt.show()
