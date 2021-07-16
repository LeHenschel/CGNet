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


if __name__ == "__main__":
    import pandas as pd
    import seaborn as sns
    import matplotlib as mpl
    plt.style.use("seaborn")
    df = pd.read_csv("benchmark.csv")
    df['Type'] = ["+ " + x if y is True else "- MST" for (x, y) in zip(df["mst_weight"], df['mst'])]
    df['best_error'] = df["best_error"] * 100
    mean_none_mst = df["train_time"].loc[(df["Type"] == "- MST")].mean()
    df["train_time"] = df["train_time"] / mean_none_mst

    prop = mpl.font_manager.FontProperties(family='sans-serif', size=6, weight="bold", style="normal")
    tick_prop = mpl.font_manager.FontProperties(family='sans-serif', size=6, weight="normal", style="normal")
    fig, ax = plt.subplots(2, 1, sharex=True)
    ax1 = sns.boxplot(y="train_time", x="Type", ax=ax[0], linewidth=1.0, data=df)
    ax1.grid(b=True)
    ax1.set_xlabel("", fontproperties=prop)
    ax1.set_ylabel("Rel. Training time", fontproperties=prop)
    for label in ax1.get_xticklabels():
        label.set_fontproperties(tick_prop)

    for tick in ax1.get_xticklabels():
        tick.set_rotation(90)

    ax2 = sns.boxplot(y="best_error", x="Type", ax=ax[1], linewidth=1.0, data=df)
    ax2.grid(b=True)
    ax2.set_xlabel("", fontproperties=prop)
    ax2.set_ylabel("Best Error [% FP]", fontproperties=prop)
    for label in ax2.get_xticklabels():
        label.set_fontproperties(tick_prop)

    for tick in ax2.get_xticklabels():
        tick.set_rotation(90)
    plt.subplots_adjust(wspace=.05)

    fig.savefig("Test.svg")
    axis_width = "\\figw"
    axis_height = "\\figh"
    tpl.save("data_pd.tex", axis_width=axis_width, axis_height=axis_height)

    plt.close(fig)
