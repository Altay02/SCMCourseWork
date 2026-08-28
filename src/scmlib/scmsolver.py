import numpy as np
import pandas as pd
import cvxpy as cp
from scipy.optimize import minimize

def solve_w_pure(n_donors, v_diag, x0, x1):
    w = cp.Variable(n_donors)
    v_mat_unnormed = np.diag(v_diag)
    v_mat_normed = v_mat_unnormed / np.linalg.norm(v_mat_unnormed, ord=2)
    obj = cp.Minimize(cp.quad_form(x1 - x0 @ w, v_mat_normed))
    constraints = [w >= 0, cp.sum(w) == 1]
    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.SCS, eps=1e-6)
    return w.value

def objective_v_fixed(v_diag, n_donors, Y_0_pre, Y_1_pre, x0, x1):
    #v_diag_normed = v_diag / np.linalg.norm(v_diag, ord=2)
    w_star = solve_w_pure(n_donors, v_diag, x0, x1)
    if w_star is None: return 1e10
    error = np.sqrt(np.mean(np.square(Y_1_pre - Y_0_pre @ w_star)))
    return error

def scm_solve(Y_0, Y_0_pre, Y_1_pre, x0, x1):
    n_features = x0.shape[0]
    n_donors = x0.shape[1]
    print(f"n_features = x0.shape[0]: {x0.shape[0]}, n_donors = x0.shape[1]: {x0.shape[1]}")
    v_start = np.ones(n_features) / n_features

    # Запуск вложенной оптимизации
    res = minimize(
        objective_v_fixed,
        v_start,
        args=(n_donors, Y_0_pre, Y_1_pre, x0, x1),
        method='L-BFGS-B',
        bounds=[(1e-6, None)] * n_features,
        #options={'maxiter': 50}
        options={'maxiter': 30, 'disp': True}
    )

    # Финальные веса
    V_diag_final = res.x / np.linalg.norm(res.x, ord=2)
    W_opt_final = solve_w_pure(n_donors, V_diag_final, x0, x1)
    Y_1_synth = Y_0 @ W_opt_final
    
    print("Оптимизация V и W успешно завершена(50 iterations). returned: {V, W, Y_1_synth}\n")
    return {"V": V_diag_final, "W": W_opt_final,"Y_1_synth": Y_1_synth}
    