from functions import *
import os
import time

# Purpose:
# This script is used solve numTrials many robust
# and Markowitz portfolio optimization problems for a range
# of kappa and n values.

start = time.time()

seed = 0
numTrials = 10

kappa_n_range = np.linspace(1e-1, 1.0, 10)
samples_array = np.array([1.0, 3.0, 6.0, 12.0, 24.0, 120.0])

numSamples = len(samples_array)
numKappa = len(kappa_n_range)

robust_returns = np.zeros([numSamples, numKappa])
markowitz_returns = np.zeros([numSamples, numKappa])

assets = range(11)

for i in range(numSamples):
    result = solveProblems(seed, numTrials, samples_array[i], -1, assets)
    for j in range(numKappa):
        markowitz_returns[i][j] = result

for i in range(len(samples_array)):
    for j in range(len(kappa_n_range)):
        print(j)
        kappa = kappa_n_range[j] / samples_array[i]
        result = solveProblems(seed, numTrials, samples_array[i], kappa, assets)
        robust_returns[i][j] = result


# get the directory of the project
project_directory = os.getcwd()
folder_name = "outputs/kappa_n"

file_name_markowitz = "markowitz_results_kn.txt"
save_directory_markowitz = os.path.join(
    project_directory, folder_name, file_name_markowitz
)
np.savetxt(save_directory_markowitz, markowitz_returns, delimiter="\t")

file_name_robust = "robust_results_kn.txt"
save_directory_robust = os.path.join(project_directory, folder_name, file_name_robust)
np.savetxt(save_directory_robust, robust_returns, delimiter="\t")

# print time elapsed
end = time.time()
time_elapsed = end - start
print(time_elapsed)

