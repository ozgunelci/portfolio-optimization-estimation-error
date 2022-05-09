import time
from functions import *

# Purpose:
# This script is used to draw Robust and Markowitz frontier.
# It requires the output files from main_frontier_robust_solve.py
# for the specified input setting.

start = time.time()

# input
seed = 0
numSamples = 24
numTrials = 100

experiment_name = "_figure_robust"
experiment_name += (
    "_error_cov_0004"
    + "_seed_"
    + str(seed)
    + "_sample_"
    + str(numSamples)
    + "_trial_"
    + str(numTrials)
)

# get the directory of the project
project_directory = os.getcwd()
folder_name = "outputs/frontier_garlappi"

file_name_markowitz_actual = "markowitz_actual" + experiment_name + ".txt"
file_name_markowitz_estimated = "markowitz_estimated" + experiment_name + ".txt"
file_name_robust_actual = "robust_actual" + experiment_name + ".txt"
file_name_robust_estimated = "robust_estimated" + experiment_name + ".txt"

markowitz_actual_directory = os.path.join(
    project_directory, folder_name, file_name_markowitz_actual
)
markowitz_estimated_directory = os.path.join(
    project_directory, folder_name, file_name_markowitz_estimated
)
robust_actual_directory = os.path.join(
    project_directory, folder_name, file_name_robust_actual
)
robust_estimated_directory = os.path.join(
    project_directory, folder_name, file_name_robust_estimated
)

data = read_burak_data()
mu = data.trueExpectedReturn
mu = np.array([mu])

# v_begin = 0.00125
# v_end = 0.00400
# precision = 0.00025
# num_v = (v_end - v_begin) / precision
# v_array = np.linspace(v_begin, v_end, int(num_v) + 1)

gamma_array_begin = 2
gamma_array_end = -7
gamma_array_index = np.linspace(gamma_array_begin,gamma_array_end, gamma_array_begin - gamma_array_end + 1)
gamma_array = np.logspace(gamma_array_begin,gamma_array_end, gamma_array_begin - gamma_array_end + 1, base = 2.0)

# # specify error_cov
# kappa = 1
# error_covariance_matrix = kappa * kappa * np.ones(11) * 0.0001

assets = range(11)

# obtain the frontiers
true_frontier = getFrontier_garlappi(assets, gamma_array, mu, -1)
true_frontier = true_frontier.actual_frontier

[equal_return, equal_risk] = getEqualFrontier(assets, gamma_array)
equal_frontier = [equal_return, equal_return]
v_equal_frontier = [equal_risk, gamma_array[-1]]

markowitz_actual_frontier = np.loadtxt(markowitz_actual_directory)
markowitz_estimated_frontier = np.loadtxt(markowitz_estimated_directory)
robust_actual_frontier = np.loadtxt(robust_actual_directory)
robust_estimated_frontier = np.loadtxt(robust_estimated_directory)

# draw the figure
drawFrontiers_garlappi(
    experiment_name,
    true_frontier,
    gamma_array_index,
    markowitz_estimated_frontier,
    markowitz_actual_frontier,
    equal_frontier,
    v_equal_frontier,
    robust_estimated_frontier,
    robust_actual_frontier,
)

end = time.time()
time_elapsed = end - start
print(time_elapsed)
