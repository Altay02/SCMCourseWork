import numpy as np
import pandas as pd
from scipy.optimize import fmin_slsqp, minimize
from matplotlib import pyplot as plt

def basic_dataprep(predictors_matrix, outcomes_matrix, treated_unit,
                   control_units, preintervention_years, years_plot):
    # check if data types match expectations.
    if not type(predictors_matrix) == pd.core.frame.DataFrame:
        raise NameError("Error 1")
    elif not type(outcomes_matrix) == pd.core.frame.DataFrame:
        raise NameError("Error 2")
    elif not type(treated_unit) == str:
        raise NameError("Error 3")
    elif not type(control_units) == list:
        raise NameError("Error 4")
    # elif not type(predictors_optimize) == list:
    #     raise NameError("Error 5")
    elif not type(preintervention_years) == list:
        raise NameError("Error 6")
    elif not type(years_plot) == list:
        raise NameError("Error 7")

    # if the list of controls contains the treated unit, remove treated unit.
    while treated_unit in control_units:
        control_units.remove(treated_unit)

    # check for empty lists
    if len(control_units) == 0 or len(preintervention_years) == 0:
           raise NameError("Error 8")

    # check for whether there are repeated control units, or more controls
    # than columns in the input matrices.
    if len(control_units) >= predictors_matrix.shape[1] or len(control_units) >= outcomes_matrix.shape[1]:
           raise NameError("Error 9")

    X1 = predictors_matrix[treated_unit]
    X0 = predictors_matrix.copy()
    del X0[treated_unit]

    Y1 = outcomes_matrix.loc[years_plot][treated_unit]
    Y0 = outcomes_matrix.loc[years_plot][control_units]
    Y1_pre = outcomes_matrix.loc[preintervention_years][treated_unit]
    Y0_pre = outcomes_matrix.loc[preintervention_years][control_units]

    return X0, X1, Y0_pre, Y1_pre, Y0, Y1

def predictors_errorByW(w, v, x0, x1):
#   return synth_x1_error_vector_normByV = (x1 - x0 @ w).T @ V @ (x1 - x0 @ w)
    k = len(x1)
    V_matrix = np.zeros((k,k))
    np.fill_diagonal(V_matrix, v)
    synth_x1_error_vector = x1 - x0 @ w
    # errors = x1 - predictions
    # weighted_errors = np.dot(error_vector.transpose(), V_matrix)
    # weighted_rss = np.dot(weighted_errors,error_vector).item(0)
    synth_x1_error_vector_normByV = synth_x1_error_vector.T @ V_matrix @ synth_x1_error_vector
    return synth_x1_error_vector_normByV

def v_rss(W, Y0_pre, Y1_pre): 
#   return synth_Y1_pre_error_norm = (Y1_pre - Y0_pre @ W(V))'*(Y1_pre - Y0_pre @ W(V))
    synth_Y1_pre = Y0_pre @ W
    synth_Y1_pre_error_vector = Y1_pre - synth_Y1_pre
    synth_Y1_pre_error_norm = sum(synth_Y1_pre_error_vector**2)
    return synth_Y1_pre_error_norm

def w_constraint(w, v, x0, x1):
    return np.sum(w) - 1

def get_w(w, v, x0, x1):
    result = fmin_slsqp(predictors_errorByW, w, f_eqcons=w_constraint, bounds=[(0.0, 1.0)]*len(w),
             args=(v, x0, x1), disp=False, full_output=True)
    weights = result[0]
    return weights

def outcome_errorByV(v, w, x0, x1, Y0_pre, Y1_pre): # return mspe = norm(Y1_pre - Y0_pre @ W(V))
    optimal_W_for_V = fmin_slsqp(predictors_errorByW, w, f_eqcons=w_constraint, bounds=[(0.0, 1.0)]*len(w),
             args=(v, x0, x1), disp=False, full_output=True)[0]
    mspe = v_rss(optimal_W_for_V, Y0_pre, Y1_pre) # mspe = norm(Y1_pre - Y0_pre @ W(V))
    return mspe

def get_v(v, w, x0, x1, Y0_pre, Y1_pre): # return argmin(V) = norm(Y1_pre - Y0_pre @ W(V))
    result = minimize(outcome_errorByV, v, args=(w, x0, x1, Y0_pre, Y1_pre), bounds=[(0.0, 1.0)]*len(v))
    optimal_V = result.x
    return optimal_V

def get_estimate(x0, x1, z0, z1, z2):
    k,j = x0.shape[0], x0.shape[1]
    v = np.array([1.0/k]*k)
    w = np.array([1.0/j]*j)
    predictors_v = get_v(v, w, x0, x1, z0, z1)
    donor_w = get_w(w, predictors_v, x0, x1)
    treated_synth = np.dot(z2,donor_w)
    return treated_synth, predictors_v, donor_w

def synth_tables(predictors_matrix, outcomes_matrix, treated_unit, control_units,
                preintervention_years, years_plot):
  
    """подсказка, что тут за Z матрицы
    X1 = predictors_matrix[treated_unit]
    del predictors_matrix[treated_unit]
    X0 = predictors_matrix

    Y1 = outcomes_matrix.loc[years_plot][treated_unit]
    Y0 = outcomes_matrix.loc[years_plot][control_units]
    Y1_pre = outcomes_matrix.loc[preintervention_years][treated_unit]
    Y0_pre = outcomes_matrix.loc[preintervention_years][control_units]
    """
    X0, X1, Y0_pre, Y1_pre, Y0, Y1 = basic_dataprep(predictors_matrix, outcomes_matrix,
        treated_unit, control_units, preintervention_years, years_plot)

    synth_Y1, matrix_V, vector_W = get_estimate(X0, X1, Y0_pre, Y1_pre, Y0)


    synth_predictors = X0 @ vector_W
    predictors_table = pd.DataFrame({'Synthetic':synth_predictors, 'Actual': X1},
                                    index=X1.index)

    # synth_outcomes = Y0 @ vector_W 
    # дублирующий расчет synth_treated_outcome = Y0 @ vector_W
    outcomes_table = pd.DataFrame({'Synthetic':synth_Y1, 'Actual':Y1},
                                  index=Y1.index)

    print("Predictors Table")
    print("---")
    print(predictors_table)
    print(" ")
    print("Outcomes Table")
    print("---")
    print(outcomes_table)
    print(" ")
    print("Predictors' Weights V")
    print("---")
    print(matrix_V)
    print(" ")
    print("Donors' Weights W")
    print("---")
    print(vector_W)

    return synth_Y1, Y1, matrix_V, vector_W, predictors_table, outcomes_table