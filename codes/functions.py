import numpy as np
import pandas as pd
from pyomo.environ import *
import os


def read_burak_data():
    class Data:
        # read data from the Excel file
        return_data = pd.read_excel(
            "inputs/data_burak.xlsx",
            sheet_name="true_return",
            index_col=None,
            header=None,
        )
        cov_data = pd.read_excel(
            "inputs/data_burak.xlsx",
            sheet_name="true_covariance",
            index_col=None,
            header=None,
        )

        trueExpectedReturn = return_data.values.tolist()
        trueExpectedReturn = trueExpectedReturn[0]

        trueCovarianceReturn = cov_data.values.tolist()

        numAssets = len(trueExpectedReturn)
        assets = list(range(0, numAssets))

    return Data


def solve_portfolio(mu, Sigma, v, errorCov):
    # mu is an one-dimensional array
    # sigma is a two-dimensional array
    # v is a scalar
    # errorCov -1 means the model is not robust.
    # if errorCov is an one-dimensional array, then the errorCov is diagonal.
    # this problem does not feature a kappa term, kappa should be embedded
    # to the error covariance matrix

    if type(errorCov) == int:
        modelRobust = False
    else:
        modelRobust = True
        if errorCov.ndim == 1:
            errorCov = np.diag(errorCov)

    assets = list(range(len(mu)))

    # initialize pyomo model
    model = ConcreteModel()

    # add the decision variables
    def variable_bounds(model, i):
        return (0, 1)

    def variable_initialize(model, i):
        return 0.05

    model.x = Var(assets, within=Reals, initialize=variable_initialize)

    # bounds = variable_bounds,

    # add the objective function
    def objective_rule(model):
        expr = sum(mu[i] * model.x[i] for i in assets)
        if modelRobust:
            expr -= sqrt(
                sum(
                    errorCov[i][j] * model.x[i] * model.x[j]
                    for i in assets
                    for j in assets
                )
            )
        return expr

    model.obj = Objective(rule=objective_rule, sense=maximize)

    # add the constraints
    model.total_asset = Constraint(expr=sum(model.x[i] for i in assets) == 1)

    model.variance = Constraint(
        expr=sum(Sigma[i][j] * model.x[i] * model.x[j] for i in assets for j in assets)
        <= v
    )

    def nonNegativity_rule(model, i):
        return model.x[i] >= 0

    model.nonNegativity = Constraint(assets, rule=nonNegativity_rule)

    model.dual = Suffix(direction=Suffix.IMPORT)

    # specify the solver
    solver_name = "ipopt"
    solver = SolverFactory(solver_name)
    if solver_name == "ipopt":
        # a list of ipopt options: https://www.coin-or.org/Bonmin/option_pages/options_list_ipopt.html
        solver.options["max_cpu_time"] = 60

    # solve the optimization model
    results = solver.solve(model)
    model.solutions.store_to(results)

    class Results:
        # if solver_name == 'ip_opt':

        solution_status = results.Solution.Status
        optimal_value = results.Solution.objective["obj"]["Value"]
        active_variance = sum(
            value(model.x[i]) * value(model.x[j]) * Sigma[i][j]
            for i in assets
            for j in assets
        )

        x_val = np.zeros(len(assets))
        for i in assets:
            x_val[i] = value(model.x[i])

        lambda1_val = model.dual.get(model.variance)
        lambda2_val = model.dual.get(model.total_asset)
        lambda3_val = np.zeros(len(assets))
        for i in assets:
            lambda3_val[i] = model.dual.get(model.nonNegativity[i])

    return Results


def getMuHat(numTrials, numSamples, seed):
    data = read_burak_data()

    mu = data.trueExpectedReturn
    sigma = data.trueCovarianceReturn

    # multiplier = 1 / sqrt((numSamples))
    multiplier = 1 / numSamples

    sigma = np.multiply(sigma, multiplier)

    np.random.seed(seed)
    muHat = np.random.multivariate_normal(mu, sigma, numTrials)

    return muHat


def drawFrontiers(
    name,
    true_frontier,
    v_array,
    markowitz_estimated_frontier,
    markowitz_actual_frontier,
    equal_frontier,
    v_equal_frontier,
    robust_estimated_frontier,
    robust_actual_frontier,
):
    import matplotlib.pyplot as plt

    f = plt.figure()
    ax = f.add_subplot(111)

    plt.plot(
        v_equal_frontier,
        equal_frontier,
        label="equal",
        color="gray",
        linestyle="dashed",
    )

    plt.plot(v_array, true_frontier, label="true", linewidth=2)
    plt.plot(
        v_array,
        markowitz_actual_frontier,
        label="actual_markowitz",
        marker="o",
        linewidth=1.5,
        color="orange",
        markersize=4,
    )
    # plt.plot(v_array, markowitz_estimated_frontier, label='estimated_markowitz', marker='o', linewidth=1.5, color='red',
    #          markersize=4)

    if type(robust_estimated_frontier) != int:
        plt.plot(
            v_array,
            robust_actual_frontier,
            label="actual_robust",
            marker="o",
            linewidth=1.5,
            color="yellowgreen",
            markersize=4,
        )
        # plt.plot(v_array, robust_estimated_frontier, label='estimated_robust', marker='o', linewidth=1.5,
        #          color='purple', markersize=4)

    plt.legend()

    plt.xlabel("variance")
    plt.ylabel("expected return")

    plt.xlim(0.00115, 0.0041)
    #    plt.ylim(0.01, 0.03)
    plt.ylim(0.012, 0.022)

    # plt.title('Line graph!')

    ratio = 0.75
    xleft, xright = ax.get_xlim()
    ybottom, ytop = ax.get_ylim()
    # the abs method is used to make sure that all numbers are positive
    # because x and y axis of an axes maybe inversed.
    ax.set_aspect(abs((xright - xleft) / (ybottom - ytop)) * ratio)

    # plt.show()

    # get the directory of the project
    project_directory = os.getcwd()
    folder_name = "outputs\\frontier"
    figure_name = name + ".pdf"
    save_directory = os.path.join(project_directory, folder_name, figure_name)

    f.savefig(save_directory, bbox_inches="tight")


def getEqualFrontier(assets, v_array):
    data = read_burak_data()

    mu = data.trueExpectedReturn
    sigma = data.trueCovarianceReturn

    numAssets = len(assets)
    allocation = 1 / numAssets

    risk = sum(sigma[i][j] * allocation * allocation for i in assets for j in assets)
    expected_return = sum(mu[i] * allocation for i in assets)

    return [expected_return, risk]


def getFrontier(assets, v_array, muHat, errorCov):
    actual = np.zeros(len(v_array))
    estimated = np.zeros(len(v_array))

    data = read_burak_data()

    mu = data.trueExpectedReturn
    sigma = data.trueCovarianceReturn

    numTrials = len(muHat)

    for t in range(numTrials):
        print(t)
        for v in range(len(v_array)):
            result = solve_portfolio(muHat[t], sigma, v_array[v], errorCov)
            estimated[v] += result.optimal_value
            actual[v] += sum(mu[i] * result.x_val[i] for i in assets)

    class results:
        actual_frontier = actual / numTrials
        estimated_frontier = estimated / numTrials

    return results


def solveProblems(seed, numTrials, numSamples, kappa, assets):
    # get samples
    muHat = getMuHat(numTrials, numSamples, seed)
    muHat_s = muHat[:, assets]

    numAssets = len(assets)

    v_array = [0.002]

    # file name
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

    project_directory = os.getcwd()
    folder_name = "outputs/kappa_n"

    # obtain solutions
    data = read_burak_data()
    mu = data.trueExpectedReturn
    sigma = data.trueCovarianceReturn
    mu = np.array(mu)
    sigma = np.array(sigma)
    mu_s = mu[assets]
    sigma_s = sigma[:, assets]
    sigma_s = sigma_s[assets, :]

    kappa_square = kappa * kappa
    errorCov = np.ones(numAssets) * kappa_square

    if kappa < 0:
        markowitz_results = getFrontier_points(
            assets, v_array, muHat_s, -1, mu_s, sigma_s
        )
        markowitz_actual_points = markowitz_results.actual_points
        markowitz_actual_points = np.transpose(markowitz_actual_points)
        actual = sum(markowitz_actual_points) / numTrials

        file_name = "markowitz_" + experiment_name + ".txt"
        save_directory_markowitz = os.path.join(
            project_directory, folder_name, file_name
        )
        np.savetxt(save_directory_markowitz, markowitz_actual_points)

    else:
        robust_results = getFrontier_points(
            assets, v_array, muHat_s, errorCov, mu_s, sigma_s
        )
        robust_actual_points = robust_results.actual_points
        robust_actual_points = np.transpose(robust_actual_points)
        actual = sum(robust_actual_points) / numTrials

        file_name = "robust_" + experiment_name + ".txt"
        save_directory_robust = os.path.join(project_directory, folder_name, file_name)
        np.savetxt(save_directory_robust, robust_actual_points)

    return actual


def getFrontier_points(assets, v_array, muHat, errorCov, mu, sigma):
    numTrials = len(muHat)

    actual = np.zeros([len(v_array), numTrials])
    estimated = np.zeros([len(v_array), numTrials])

    solution = np.zeros([len(v_array), numTrials, len(mu)])

    for t in range(numTrials):
        # print(t)
        for v in range(len(v_array)):
            result = solve_portfolio(muHat[t], sigma, v_array[v], errorCov)
            estimated[v][t] = result.optimal_value
            actual[v][t] = sum(mu[i] * result.x_val[i] for i in assets)

            for i in range(len(mu)):
                solution[v][t][i] = result.x_val[i]

    class results:
        actual_points = actual
        estimated_points = estimated
        optimal_solution = solution

    return results


def perform_bootstrap(no_bootstrap, sample_array):
    num_sample = len(sample_array)
    boot_distribution = np.zeros(no_bootstrap)

    for k in range(no_bootstrap):
        boot_sample = np.random.choice(sample_array, replace=True, size=num_sample)
        boot_statistic = np.average(boot_sample)
        boot_distribution[k] = boot_statistic

    class Results:
        bootstrap_distribution = boot_distribution
        bootstrap_error = np.std(boot_distribution)
        bootstrap_CI = np.percentile(boot_distribution, [2.5, 97.5])

    return Results


def perform_bootstrap_ratio(
    no_bootstrap, sample_array_markowitz, sample_array_robust
):
    # this is a much faster implementation of the previous one.
    true_return = 0.01534932
    num_sample = len(sample_array_markowitz)
    boot_distribution = np.zeros(no_bootstrap)

    for k in range(no_bootstrap):
        np.random.seed(k)
        average_markowitz = (
            sum(np.random.choice(sample_array_markowitz, replace=True, size=num_sample))
            / num_sample
        )

        np.random.seed(k)
        average_robust = (
            sum(np.random.choice(sample_array_robust, replace=True, size=num_sample))
            / num_sample
        )

        boot_statistic = ((average_robust - average_markowitz) * 100) / (
            true_return - average_markowitz
        )
        boot_distribution[k] = boot_statistic

    class Results:
        bootstrap_distribution = boot_distribution
        bootstrap_error = np.std(boot_distribution)
        bootstrap_CI = np.percentile(boot_distribution, [2.5, 97.5])

    return Results

