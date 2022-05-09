import time
from functions import *

# Purpose:
# This script solves numTrials many portfolio optimization problems
# for the specified input setting.
# It saves the optimal objective function values.
# The output is used to draw Markowitz frontier.

start = time.time()

# input
seed = 0
numSamples = 24
numTrials = 100

experiment_name = "_figure_markowitz"
experiment_name += (
    "_seed_" + str(seed) + "_sample_" + str(numSamples) + "_trial_" + str(numTrials)
)

muHat = getMuHat(numTrials, numSamples, seed)

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

assets = range(11)

# obtain the frontiers
markowitz_results = getFrontier_garlappi(assets, gamma_array, muHat, -1)
markowitz_estimated_frontier = markowitz_results.estimated_frontier
markowitz_actual_frontier = markowitz_results.actual_frontier

# save the results to a txt file
# get the directory of the project
project_directory = os.getcwd()
folder_name = "outputs/frontier_garlappi"

# markowitz actual
file_name = "markowitz_actual" + experiment_name + ".txt"
save_directory = os.path.join(project_directory, folder_name, file_name)
np.savetxt(save_directory, markowitz_actual_frontier)

# markowitz estimated
file_name = "markowitz_estimated" + experiment_name + ".txt"
save_directory = os.path.join(project_directory, folder_name, file_name)
np.savetxt(save_directory, markowitz_estimated_frontier)


end = time.time()
time_elapsed = end - start
print(time_elapsed)
