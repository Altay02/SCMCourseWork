import numpy as np
import pandas as pd
import cvxpy as cp
from scipy.optimize import minimize

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

def get_w(n_donors, v_diag, x0, x1):
    # print("n_donors will be: ", n_donors)
    w = cp.Variable(n_donors)
    v_mat_unnormed = np.diag(v_diag)
    #v_mat_normed = v_mat_unnormed / np.linalg.norm(v_mat_unnormed, ord=2)
    #if (x0.shape[1] != w.shape[0]):
    #print("type: x0, w:", type(x0), type(w), sep=' ')
    #print("x0.shape, w.shape:", x0.shape, w.shape, sep=' ')
    obj = cp.Minimize(cp.quad_form(x1 - x0 @ w, v_mat_unnormed))
    constraints = [w >= 0, cp.sum(w) == 1]
    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.SCS, eps=1e-6)
    return w.value

def mspe_by_v(v_diag, n_donors, Y_0_pre, Y_1_pre, x0, x1):
    #v_diag_normed = v_diag / np.linalg.norm(v_diag, ord=2)
    w_star = get_w(n_donors, v_diag, x0, x1)
    if w_star is None: return 1e10
    mspe = np.sum(np.square(Y_1_pre - Y_0_pre @ w_star))
    return mspe

def scm_solve(Y_0, Y_0_pre, Y_1_pre, x0, x1):
    n_features = x0.shape[0]
    n_donors = x0.shape[1]
    print(f"n_features = x0.shape[0]: {x0.shape[0]}, n_donors = x0.shape[1]: {x0.shape[1]}")
    v_start = np.ones(n_features) / n_features

    # Запуск вложенной оптимизации
    res = minimize(
        mspe_by_v,
        v_start,
        args=(n_donors, Y_0_pre, Y_1_pre, x0, x1),
        method='L-BFGS-B',
        bounds=[(1e-6, None)] * n_features,
        #options={'maxiter': 50}
        options={'maxiter': 20, 'disp': True}
    )

    # Финальные веса
    V_diag_final = res.x / np.linalg.norm(res.x, ord=2)
    W_opt_final = get_w(n_donors, V_diag_final, x0, x1)
    Y_1_synth = Y_0 @ W_opt_final
    
    print("Оптимизация V и W успешно завершена(50 iterations). returned: {V, W, Y_1_synth}\n")
    return {"V": V_diag_final, "W": W_opt_final,"Y_1_synth": Y_1_synth}
    
