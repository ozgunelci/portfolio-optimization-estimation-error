import time
from functions_new import *
import os

# USED FOR THE PAPER

# Purpose:
# This file solves numTrials many portfolio optimization problems
# for the specified input setting.
# It saves the optimal objective function values as well as the
# optimal solutions.
# The output is used to draw histogram files.

start = time.time()

# input
seed = 0
numSamples = 120
numTrials = 100
kappa = (
    0.5 / numSamples
)  # this is not used for the Markowitz model - kept here to align with the robust model.
v_array = [0.002]

# experiment name
v = v_array[0]
kappa_dec = str(kappa - int(kappa))[2:]
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

# specify error matrix
kappa_square = kappa * kappa
errorCov = np.ones(11) * kappa_square

# obtain solutions
robust_results = getFrontier_points_final(assets, v_array, muHat, errorCov, mu, sigma)
robust_estimated_points = robust_results.estimated_points
robust_actual_points = robust_results.actual_points
robust_actual_points = np.transpose(robust_actual_points)
print("robust actual = " + str(sum(robust_actual_points) / numTrials))

# get the directory of the project
project_directory = os.getcwd()
folder_name = "outputs/histogram"

# save the actual points to a txt file
file_name = "robust_actual_" + experiment_name + ".txt"
save_directory = os.path.join(project_directory, folder_name, file_name)
np.savetxt(save_directory, robust_actual_points)

# save the solution to a txt file
solution = robust_results.optimal_solution[0]
solution = np.transpose(solution)
file_name = "robust_solution_" + experiment_name + ".txt"
save_directory = os.path.join(project_directory, folder_name, file_name)
np.savetxt(save_directory, solution)

# print time elapsed
end = time.time()
time_elapsed = end - start
print(time_elapsed)
