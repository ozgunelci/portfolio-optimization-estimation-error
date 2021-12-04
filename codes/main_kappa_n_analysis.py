# This script is used to run the experiments.

from functions_new import *
import os

seed = 0
numTrials = 10000

# kappa_n_range = np.linspace(1e-1, 1.0, 10)
kappa_n_range = np.array([0.45, 0.55])
samples_array = np.array([1.0, 3.0, 6.0, 12.0, 24.0, 120.0])

numSamples = len(samples_array)
numKappa = len(kappa_n_range)

robust_returns = np.zeros([numSamples, numKappa])
markowitz_returns = np.zeros([numSamples, numKappa])

assets = range(11)

# for i in range(numSamples):
#     result = solveProblems_final(seed, numTrials, samples_array[i], -1, assets)
#     for j in range(numKappa):
#         markowitz_returns[i][j] = result

for i in range(len(samples_array)):
    for j in range(len(kappa_n_range)):
        print(j)
        kappa = kappa_n_range[j] / samples_array[i]
        result = solveProblems_final(seed, numTrials, samples_array[i], kappa, assets)
        robust_returns[i][j] = result


# get the directory of the project
project_directory = os.getcwd()
folder_name = 'outputs_kappa_n_new'

# file_name_markowitz = 'markowitz_results_kn.txt'
# save_directory_markowitz = os.path.join(project_directory, folder_name, file_name_markowitz)
# np.savetxt(save_directory_markowitz, markowitz_returns, delimiter='\t')

file_name_robust = 'robust_results_kn.txt'
save_directory_robust = os.path.join(project_directory, folder_name, file_name_robust)
np.savetxt(save_directory_robust, robust_returns, delimiter='\t')

