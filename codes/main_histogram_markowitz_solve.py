import time
from functions import *
import os

# Purpose:
# This script solves numTrials many portfolio optimization problems
# for the specified input setting.
# It saves the optimal objective function values as well as the
# optimal solutions.
# The output is used to draw histogram files.

start = time.time()

# input
seed = 0
numSamples = 3
numTrials = 10000
kappa = (
    0.5 / numSamples
)  # this is not used for the Markowitz model - kept here to align with the robust model.
v_array = [0.002]

# experiment name
v = v_array[0]
kappa_dec = str(round(kappa - int(kappa), 10))[2:]
v_dec = str(v - int(v))[2:]
experiment_name = (
    "kappa_"
    + kappa_dec
    + "_trial_"
    + str(numTrials)
    + "_sample_"
    + str(numSamples)
    + "_risk_"
    + v_dec
)

# get samples
muHat = getMuHat(numTrials, numSamples, seed)

# obtain data
assets = range(11)
data = read_burak_data()
mu = data.trueExpectedReturn
sigma = data.trueCovarianceReturn

# obtain solutions
markowitz_results = getFrontier_points(assets, v_array, muHat, -1, mu, sigma)
markowitz_estimated_points = markowitz_results.estimated_points
markowitz_actual_points = markowitz_results.actual_points
markowitz_actual_points = np.transpose(markowitz_actual_points)
print("markowitz actual = " + str(sum(markowitz_actual_points) / numTrials))

# get the directory of the project
project_directory = os.getcwd()
folder_name = "outputs/histogram"

# save the actual points to a txt file
file_name = "markowitz_actual_" + experiment_name + ".txt"
save_directory = os.path.join(project_directory, folder_name, file_name)
np.savetxt(save_directory, markowitz_actual_points)

# save the solution to a txt file
solution = markowitz_results.optimal_solution[0]
solution = np.transpose(solution)
file_name = "markowitz_solution_" + experiment_name + ".txt"
save_directory = os.path.join(project_directory, folder_name, file_name)
np.savetxt(save_directory, solution)

# print time elapsed
end = time.time()
time_elapsed = end - start
print(time_elapsed)
