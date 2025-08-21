import numpy as np
import warnings
import itertools

def compute_certificate(
        w, 
        q, 
        w_cert,
        q_cert, 
        primal_grad, 
        linear_gap, 
        X, 
        y, 
        primal_oracle, 
        dual_oracle, 
        weight_decay, 
        fit_intercept,
        shift_cost=0.05,
        penalty="chi2",
    ):

    # update tracked quantities with maximum recency
    losses, grads = primal_oracle(w, X, y)
    avg = 0.1
    q_new = (1 - avg) * q + avg * q_cert
    w_new = (1 - avg) * w + avg * w_cert
    primal_grad_new = (1 - avg) * grads.T @ q + avg * primal_grad
    linear_gap_new = (1 - avg) * q @ (losses - grads @ w) + avg * linear_gap

    losses_new, _ = primal_oracle(w_new, X, y)
    primal_reg = 0.5 * weight_decay * (w_new[int(fit_intercept):] ** 2).sum()
    n = len(losses_new)

    if penalty == "chi2":
        dual_reg = shift_cost * n * ((q_new - 1. / n) ** 2).sum()
    elif penalty == "kl":
        dual_reg = shift_cost * (q_new * np.log(n * q_new)).sum()
    else:
        raise NotImplementedError(f"Penalty {penalty} is not implemented!")
    dual_val, _ = dual_oracle(losses_new)

    # solve primal problem
    beta = 1.0 # smoothed duality gap parameter
    u = (beta * w_new - primal_grad_new) / (weight_decay + beta)
    if fit_intercept:
        u[0] = (beta * w_new[0] - primal_grad_new[0]) / beta
    primal_val = primal_grad_new @ u + 0.5 * weight_decay * (u[int(fit_intercept):] ** 2).sum() + 0.5 * beta * ((u - w_new) ** 2).sum()

    M = max(0, primal_val + linear_gap_new)

    certificate = dual_val + primal_reg + dual_reg - M

    return certificate, w_new, q_new, primal_grad_new, linear_gap_new

def compute_regularizer_grad(w, weight_decay, fit_intercept):
    reg_grad = weight_decay * w
    if fit_intercept:
        reg_grad[0] = 0.0
    return reg_grad

def evaluate_batch_gradient_norm(w, X, y, primal_oracle, dual_oracle, weight_decay, fit_intercept):
    losses, grads = primal_oracle(w, X, y)
    dual_val, q = dual_oracle(losses)
    return np.linalg.norm(q @ grads +  compute_regularizer_grad(w, weight_decay, fit_intercept))

def select_learning_rate(opt_func, X, y, lrs=[1e-4, 3e-4, 1e-3, 3e-3, 1e-2]):
    # compute loss on small number of iterations
    with warnings.catch_warnings():
        warnings.simplefilter("ignore") # ignores convergence warnings
        loss_vals = [opt_func(lr, X, y)["metrics"]["objective_value"][-1] for lr in lrs]
    return lrs[np.argmin(loss_vals)]

def evaluate(
        it, 
        w, 
        X, 
        y, 
        fit_intercept,
        primal_first_order_oracle, 
        dual_maximization_oracle, 
        weight_decay, 
        primal_oracle_evals,
        X_test=None,
        y_test=None,
    ):
    losses = primal_first_order_oracle(w, X, y)[0]
    dual_val, q = dual_maximization_oracle(losses)
    grad_norm = evaluate_batch_gradient_norm(
        w, X, y, 
        primal_first_order_oracle, 
        dual_maximization_oracle, 
        weight_decay,
        fit_intercept
    )
    result = {
        "iter": it,
        "coupled_loss": q @ losses,
        "primal_reg": 0.5 * weight_decay * (w[int(fit_intercept):] ** 2).sum(),
        "objective_value": dual_val + 0.5 * weight_decay * (w[int(fit_intercept):] ** 2).sum(),
        "primal_oracle_evals": primal_oracle_evals,
        "gradient_norm": grad_norm,
    }
    if not (X_test is None or y_test is None):
        # save outputs so offline metrics can be computed.
        result['primal_vars'] = w.copy()
    return result


def to_dict_of_lists(lst):
    return {key: [i[key] for i in lst] for key in lst[0]}

def to_list_of_dicts(d):
    for key in d:
        if not isinstance(d[key], list):
            d[key] = [d[key]]
    return [dict(zip(d, x)) for x in itertools.product(*d.values())]