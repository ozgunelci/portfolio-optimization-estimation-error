# This script is used to evaluate the experiments.

from functions import *
import os
import time

start = time.time()

seed = 0
numTrials = 10
numBootstrap = 10000

kappa_n_range = np.linspace(1e-1, 1.0, 10)
samples_array = np.array([1.0, 3.0, 6.0, 12.0, 24.0, 120.0])

true_return = 0.01534932

v_array = [0.002]

project_directory = os.getcwd()
folder_name = "outputs/kappa_n"

# initialize arrays
robust_markowitz_difference_errors = np.zeros([len(samples_array), len(kappa_n_range)])

markowitz_errors = np.zeros([len(samples_array), len(kappa_n_range)])
robust_errors = np.zeros([len(samples_array), len(kappa_n_range)])
robust_markowitz_ratio = np.zeros([len(samples_array), len(kappa_n_range)])


for i in range(len(samples_array)):
    for j in range(len(kappa_n_range)):
        print(j)

        numSamples = samples_array[i]
        kappa = kappa_n_range[j] / samples_array[i]

        # file name
        v = v_array[0]
        kappa_dec = str(round(kappa - int(kappa), 10))[2:]
        v_dec = str(v - int(v))[2:]

        # Markowitz
        experiment_name = (
            "kappa_"
            + "_trial_"
            + str(numTrials)
            + "_sample_"
            + str(numSamples)
            + "_risk_"
            + v_dec
        )
        file_name = "markowitz_" + experiment_name + ".txt"
        save_directory_markowitz = os.path.join(
            project_directory, folder_name, file_name
        )
        markowitz_actual_points = np.loadtxt(save_directory_markowitz)

        # Robust
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
        file_name = "robust_" + experiment_name + ".txt"
        save_directory_robust = os.path.join(project_directory, folder_name, file_name)
        robust_actual_points = np.loadtxt(save_directory_robust)

        result = perform_bootstrap_ratio(
            numBootstrap, markowitz_actual_points, robust_actual_points
        )
        robust_markowitz_difference_errors[i][j] = result.bootstrap_error

        result = perform_bootstrap(numBootstrap, markowitz_actual_points)
        markowitz_errors[i][j] = result.bootstrap_error

        result = perform_bootstrap(numBootstrap, robust_actual_points)
        robust_errors[i][j] = result.bootstrap_error


# write the results
project_directory = os.getcwd()
folder_name = "outputs/kappa_n"
file_name = "bootstrap_error_results_kn.txt"
save_directory = os.path.join(project_directory, folder_name, file_name)

file = open(save_directory, "a")

file.write("new_experiments \n")

file.write("robust_markowitz_difference_errors\n")
np.savetxt(file, robust_markowitz_difference_errors, delimiter="\t")
file.write("\n")
file.write("\n")

file.write("markowitz_errors\n")
np.savetxt(file, markowitz_errors, delimiter="\t")
file.write("\n")
file.write("\n")

file.write("robust_errors\n")
np.savetxt(file, robust_errors, delimiter="\t")
file.write("\n")
file.write("\n")

file.close()

# print time elapsed
end = time.time()
time_elapsed = end - start
print(time_elapsed)
