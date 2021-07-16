import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import csv
import tikzplotlib as tpl

def convert_str_to_bool(arr):
    new = []
    false_str = "False"
    true_str = "True"
    for elem in arr:
        if elem == false_str or elem == false_str.upper() or elem == false_str.lower():
            new.append(False)
        if elem == true_str or elem == true_str.upper() or elem == true_str.lower():
            new.append(True)

    return np.array(new)

mst_used = []
mst_weight = []
run_time = []
best_error = []

with open('benchmark.csv') as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    for row in csvReader:
        mst_used.append(row[0])
        mst_weight.append(row[1])
        run_time.append(row[2])
        best_error.append(row[3])

mst_used = np.array(mst_used[1:])
mst_used = convert_str_to_bool(mst_used)
mst_weight = np.array(mst_weight[1:])
run_time = np.array(run_time[1:])
best_error = np.array(best_error[1:])

no_mst_run_time = np.array([int(time) for time, used in zip(run_time, mst_used) if not used])
no_mst_err = np.array([float(berr) for berr, used in zip(best_error, mst_used) if not used])

mst_run_time = np.array([time for time, used in zip(run_time, mst_used) if used])
mst_best_error = np.array([berr for berr, used in zip(best_error, mst_used) if used])
mst_weight = np.array([weight for weight, used in zip(mst_weight, mst_used) if used])

sort_idx = np.argsort(mst_weight)

mst_weight = mst_weight[sort_idx]

mst_used = mst_used[sort_idx]
mst_run_time = mst_run_time[sort_idx]
mst_best_error = mst_best_error[sort_idx]


cost_run_time = [int(time) for time, used in zip(mst_run_time, mst_weight) if used=="cost"]
cost_err = [float(berr) for berr, used in zip(mst_best_error, mst_weight) if used=="cost"]

random_run_time = [int(time) for time, used in zip(mst_run_time, mst_weight) if used=="random"]
random_err = [float(berr) for berr, used in zip(mst_best_error, mst_weight) if used=="random"]

sum_run_time = [int(time) for time, used in zip(mst_run_time, mst_weight) if used=="sum"]
sum_err = [float(berr) for berr, used in zip(mst_best_error, mst_weight) if used=="sum"]

none_run_time = [int(time) for time, used in zip(mst_run_time, mst_weight) if used=="none"]
none_err = [float(berr) for berr, used in zip(mst_best_error, mst_weight) if used=="none"]

mean_no_mst_run_time = np.mean(no_mst_run_time)
std_no_mst_run_time = np.std(no_mst_run_time)

mean_cost_run_time = np.mean(cost_run_time)
std_cost_run_time = np.std(cost_run_time)

mean_random_run_time = np.mean(random_run_time)
std_random_run_time = np.std(random_run_time)

mean_sum_run_time = np.mean(sum_run_time)
std_sum_run_time = np.std(sum_run_time)

mean_none_run_time = np.mean(none_run_time)
std_none_run_time = np.std(none_run_time)

mean_no_mst_err = np.mean(no_mst_err)
std_no_mst_err = np.std(no_mst_err)

mean_cost_err = np.mean(cost_err)
std_cost_err = np.std(cost_err)

mean_random_err = np.mean(random_err)
std_random_err = np.std(random_err)

mean_sum_err = np.mean(sum_err)
std_sum_err = np.std(sum_err)

mean_none_err = np.mean(none_err)
std_none_err = np.std(none_err)


ms = 1.5
es = 3
fmt = "o"

_, ax = plt.subplots(2,1, sharex=True)
ax[0].errorbar(1,mean_no_mst_run_time, std_no_mst_run_time, fmt=fmt, capsize=es, ms=ms)
ax[0].errorbar(2,mean_cost_run_time, std_cost_run_time, fmt=fmt, capsize=es, ms=ms)
ax[0].errorbar(3,mean_sum_run_time, std_sum_run_time, fmt=fmt, capsize=es, ms=ms)
ax[0].errorbar(4,mean_none_run_time, std_none_run_time, fmt=fmt, capsize=es, ms=ms)
ax[0].errorbar(5,mean_random_run_time, std_random_run_time, fmt=fmt, capsize=es, ms=ms)

ylim = list(ax[0].get_ylim())
ylim[0]=0
ax[0].set_ylim(ylim)

ax[0].yaxis.grid(True, which='major')
ax[0].yaxis.grid(True, which='minor')
ax[0].set_title("Run time [s]")
ax[0].set_ylabel("[s]")

ax[1].errorbar(1,mean_no_mst_err, std_no_mst_err, fmt=fmt, capsize=es, ms=ms)
ax[1].errorbar(2,mean_cost_err, std_cost_err, fmt=fmt, capsize=es, ms=ms)
ax[1].errorbar(3,mean_sum_err, std_sum_err, fmt=fmt, capsize=es, ms=ms)
ax[1].errorbar(4,mean_none_err, std_none_err, fmt=fmt, capsize=es, ms=ms)
ax[1].errorbar(5,mean_random_err, std_random_err, fmt=fmt, capsize=es, ms=ms)

ylim = list(ax[1].get_ylim())
ylim[0]=0
ax[1].set_ylim(ylim)

ax[1].yaxis.grid(True, which='major')
ax[1].yaxis.grid(True, which='minor')
ax[1].set_title("Best Error")
# plt.show()

axis_width = "\\fig_width"
axis_height = "\\fig_height"

tpl.save("data.tex", axis_width=axis_width, axis_height=axis_height)
