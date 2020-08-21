import matplotlib.pyplot as plt
import os
import json

fx_values = [0.001, 0.01, 0.1]
metric_values_tp = []
metric_values_fn = []

for f in fx_values:
    report_name = 'performance_fpis_' + str(f)
    directory = os.path.join(os.getcwd(), 'reports', report_name)
    with open(os.path.join(directory, 'stats.json'), 'r') as file:
        stats = json.load(file)
        metric_values_tp.append(stats['tp'])
        metric_values_fn.append(stats['fn'])

plt.figure("TP and FN vs fp_error")
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
    report_name = 'performance_fnis_' + str(f)
    directory = os.path.join(os.getcwd(), 'reports', report_name)
    with open(os.path.join(directory, 'stats.json'), 'r') as file:
        stats = json.load(file)
        metric_values_tp.append(stats['tp'])
        metric_values_fn.append(stats['fn'])

plt.figure("TP and FN vs fn_error")
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

