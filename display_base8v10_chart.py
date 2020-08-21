import matplotlib.pyplot as plt
import os
import json

f_values = [0.001, 0.002, 0.003, 0.004, 0.005, 0.009, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.09, 0.1]

accuracy_values = []
for f in f_values:
    report_name = 'performance_fis' + str(f)
    directory = os.path.join(os.getcwd(), 'reports', report_name)
    with open(os.path.join(directory, 'stats.json'), 'r') as file:
        stats = json.load(file)
        accuracy_values.append(stats['accuracy'])

accuracy_values_B10 = []
for f in f_values:
    report_name = 'performance_bis10_fis' + str(f)
    directory = os.path.join(os.getcwd(), 'reports', report_name)
    with open(os.path.join(directory, 'stats.json'), 'r') as file:
        stats = json.load(file)
        accuracy_values_B10.append(stats['accuracy'])

plt.figure("Base = 8")
plt.title('Performance vs f for base 8')
plt.xscale('log')
plt.xlabel('infection rate f')
plt.ylabel('accuracy')
plt.axis([0.001, 0.1, 0, 1])
plt.tick_params(axis='both', which='both')
plt.grid(True, which='both', )
plt.plot(f_values, accuracy_values, label='Base 8')
plt.plot(f_values, accuracy_values_B10, color='lightgray', label='Base 10')
plt.legend()

plt.figure("Base = 10")
plt.title('Performance vs f for base 10')
plt.xscale('log')
plt.xlabel('infection rate f')
plt.ylabel('accuracy')
plt.axis([0.001, 0.1, 0, 1])
plt.grid(True, which='both')
plt.plot(f_values, accuracy_values, color='lightgray', label="Base 8")
plt.plot(f_values, accuracy_values_B10, label='Base 10')
plt.legend()

plt.show()
