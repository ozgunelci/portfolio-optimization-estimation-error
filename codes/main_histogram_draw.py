import numpy as np
from matplotlib import pyplot
import os

# Purpose:
# This script is used to draw the histogram figures.
# It requires the output files from
# main_histogram_markowitz_solve.py and main_histogram_robust_solve.py
# for the specified input setting.

# inputs
numTrial = 10000
n = 24
kappa = 0.5 / n
v = 0.002

# read the data
kappa_dec = str(round(kappa - int(kappa), 10))[2:]
v_dec = str(v - int(v))[2:]

# get the directory of the project
project_directory = os.getcwd()
folder_name = "outputs/histogram"

# read the files
experiment_name = (
    "kappa_"
    + kappa_dec
    + "_trial_"
    + str(numTrial)
    + "_sample_"
    + str(n)
    + "_risk_"
    + v_dec
)

file_name_markowitz = "markowitz_actual_" + experiment_name + ".txt"
file_name_robust = "robust_actual_" + experiment_name + ".txt"

markowitz_directory = os.path.join(project_directory, folder_name, file_name_markowitz)
robust_directory = os.path.join(project_directory, folder_name, file_name_robust)

markowitz_actual_points = np.loadtxt(markowitz_directory)
robust_actual_points = np.loadtxt(robust_directory)

# plot the histograms
numBins = 20
bins = np.linspace(0.0095, 0.016, 25)

markowitz_mean = str(round(np.mean(markowitz_actual_points), 6))
robust_mean = str(round(np.mean(robust_actual_points), 6))

markowitz_std = str(round(np.sqrt(np.var(markowitz_actual_points)), 6))
robust_std = str(round(np.sqrt(np.var(robust_actual_points)), 6))

markowitz_SE = str(
    round(np.sqrt(np.var(markowitz_actual_points)) / np.sqrt(numTrial), 8)
)
robust_SE = str(round(np.sqrt(np.var(robust_actual_points)) / np.sqrt(numTrial), 8))

markowitz_text = (
    "Markowitz: mean="
    + markowitz_mean
    + ", std="
    + markowitz_std
    + ", SE="
    + markowitz_SE
)
robust_text = (
    "Robust:      mean=" + robust_mean + ", std=" + robust_std + ", SE=" + robust_SE
)
plot_title = "Comparison of Markowitz and Robust Portfolios"

fig = pyplot.figure()
pyplot.style.use("seaborn-deep")

# overlapping bars
pyplot.hist(
    markowitz_actual_points,
    bins,
    alpha=0.5,
    label="Markowitz",
    density=True,
    color="darkorange",
)
pyplot.hist(
    robust_actual_points, bins, alpha=0.5, label="Robust", density=True, color="green"
)

# this should change depending on how many plots we are presenting
ax = pyplot.gca()
ax.set_ylim([0, 1100])

# side by side bars
# pyplot.hist([markowitz_actual_points, robust_actual_points], bins=numBins, label=['markowitz', 'robust'])

pyplot.legend(loc="upper left")
pyplot.title(plot_title)
pyplot.xlabel("Expected Return")
# pyplot.figtext(
#     0.1,
#     -0.05,
#     "k = "
#     + str(kappa)
#     + ", |T| = "
#     + str(numTrial)
#     + ", n = "
#     + str(n)
#     + ", v = "
#     + str(v),
# )
pyplot.figtext(0.1, -0.05, markowitz_text)
pyplot.figtext(0.1, -0.11, robust_text)
# pyplot.show()

# save the figure as pdf
file_name = experiment_name + ".pdf"
save_directory = os.path.join(project_directory, folder_name, file_name)
fig.savefig(save_directory, bbox_inches="tight")
