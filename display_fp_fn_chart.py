import matplotlib.pyplot as plt
import os
import json

fx_values = [0.001, 0.002, 0.003, 0.005, 0.008, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1]
metric_values_tp = []
metric_values_fn = []

for f in fx_values:
    report_name = 'performance_fpis_' + str(f)
    directory = os.path.join(os.getcwd(), 'reports', report_name)
    with open(os.path.join(directory, 'stats.json'), 'r') as file:
        stats = json.load(file)
        metric_values_tp.append(stats['tp']*100/(stats['tp']+stats['fn']))
        metric_values_fn.append(stats['fn']*100/(stats['tp']+stats['fn']))

plt.figure("TP and FN vs fp_error")
plt.title('True positives and false negatives vs false positive rate')
plt.xscale('log')
plt.xlabel('false positive rate fp')
plt.ylabel('% out of actual positives')
plt.tick_params(axis='both', which='both')
plt.grid(True, which='both', )
plt.plot(fx_values, metric_values_tp, label='True Positive')
plt.plot(fx_values, metric_values_fn, label='False Negative')
plt.legend()

metric_values_tp = []
metric_values_fn = []

for f in fx_values:
    report_name = 'performance_fnis_' + str(f)
    directory = os.path.join(os.getcwd(), 'reports', report_name)
    with open(os.path.join(directory, 'stats.json'), 'r') as file:
        stats = json.load(file)
        metric_values_tp.append(stats['tp']*100/(stats['tp']+stats['fn']))
        metric_values_fn.append(stats['fn']*100/(stats['tp']+stats['fn']))

plt.figure("TP and FN vs fn_error")
plt.title('True positives and false negatives vs false negative rate')
plt.xscale('log')
plt.xlabel('false negative rate fn')
plt.ylabel('% out of actual positives')
plt.tick_params(axis='both', which='both')
plt.grid(True, which='both', )
plt.plot(fx_values, metric_values_tp, label='True Positive')
plt.plot(fx_values, metric_values_fn, label='False Negative')
plt.legend()

plt.show()

