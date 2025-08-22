"""Functions for implementing the optimizers LSVRG and Prospect."""
import numpy as np
import numba
from functools import partial
import time

from .utils import evaluate, to_dict_of_lists, select_learning_rate, compute_regularizer_grad, compute_certificate
from .loss import get_primal_first_order_oracle

import warnings
from sklearn.exceptions import ConvergenceWarning

def minimize_lsvrg(
    loss,
    dual_maximization_oracle,
    X,
    y,
    fit_intercept=False,
    X_test=None,
    y_test=None,
    lr=None,
    weight_decay=0.01,
    max_iter=None,
    eval_interval=10,
    stopping_tol=1e-3,
    seed=1,
    shift_cost=0.05,
    penalty="chi2",
):
    
    tic = time.time()
    
    # initialization
    n, d = X.shape
    if lr is None:
        # define recursive function
        def opt_func(lr, X_, y_):
            return minimize_lsvrg(
                loss, 
                dual_maximization_oracle,
                X_,
                y_,
                fit_intercept=fit_intercept,
                lr=lr,
                max_iter=min(4 * n, 1600) + 1,
                eval_interval=min(4 * n, 1600),
                weight_decay=weight_decay,
            )
        lr = select_learning_rate(opt_func, X, y)

    primal_first_order_oracle = get_primal_first_order_oracle(loss)
    n_class = None
    if loss == "multinomial_cross_entropy":
        n_class = len(np.unique(y))
        d = n_class * d
        primal_first_order_oracle = partial(primal_first_order_oracle, n_class)

    if max_iter is None:
        max_iter = min(100 * n, 100000)

    np.random.seed(seed)
    rng = np.random.RandomState(seed)

    # initialize primal iterates
    w = np.zeros(d, dtype=np.float64)

    # initialize values for certificate computation
    losses, grads = primal_first_order_oracle(w, X, y)
    argsort = np.argsort(losses, kind='stable')
    dual_val, q = dual_maximization_oracle(losses)
    grad_agg = q @ grads
    w_ckpt = w.copy()
    
    w_cert = w.copy()
    q_cert = q.copy()
    primal_grad = grad_agg
    linear_gap = dual_val - grad_agg @ w

    # main loop
    metrics = []
    primal_oracle_evals = 0
    iter = 0
    for iter in range(max_iter):

        # evaluation
        if iter % eval_interval == 0:
            metrics.append(
                evaluate(
                    iter, w, X, y, 
                    fit_intercept,
                    primal_first_order_oracle, 
                    dual_maximization_oracle, 
                    weight_decay, 
                    primal_oracle_evals,
                    X_test=X_test,
                    y_test=y_test,
                )
            )
            if iter == 0:
                metrics[-1]["certificate"] = -1
            else:
                if iter == eval_interval:
                    w_cert = w.copy()
                    q_cert = q.copy()
                    loss_, grad_ = primal_first_order_oracle(w, X, y)
                    primal_grad = grad_.T @ q
                    linear_gap = q @ (loss_- grad_ @ w)
                certificate, w_cert, q_cert, primal_grad, linear_gap = compute_certificate(
                    w, 
                    q, 
                    w_cert,
                    q_cert, 
                    primal_grad, 
                    linear_gap, 
                    X, 
                    y, 
                    primal_first_order_oracle, 
                    dual_maximization_oracle, 
                    weight_decay, 
                    fit_intercept,
                    shift_cost=shift_cost,
                    penalty=penalty,
                )
                metrics[-1]["certificate"] = certificate
            if metrics[-1]["gradient_norm"] < stopping_tol * metrics[0]["gradient_norm"]:
                break
            if iter > 0 and metrics[-1]["certificate"] < stopping_tol * metrics[0]["certificate"]:
                break

        # update tables
        if iter % n == 0:
            losses, grads = primal_first_order_oracle(w, X, y)
            argsort = np.argsort(losses, kind='stable')
            dual_val, q = dual_maximization_oracle(losses)
            grad_agg = q @ grads
            w_ckpt = w.copy()

        # main update
        i = rng.randint(0, n)
        i = argsort[i]
        loss, grad = primal_first_order_oracle(w, X[i], y[i])
        loss_ckpt, grad_ckpt = primal_first_order_oracle(w_ckpt, X[i], y[i])
        primal_oracle_evals += 1
        v = n * q[i] * grad - n * q[i] * grad_ckpt + grad_agg + compute_regularizer_grad(w, weight_decay, fit_intercept)
        w -= lr * v

    if iter == max_iter:
        msg = f"LSVRG failed to converge within {max_iter} iterations! Gradient norm init: {metrics[0]['gradient_norm']}, final: {metrics[-1]['gradient_norm']}"
        warnings.warn(msg, ConvergenceWarning)

    toc = time.time()

    return {"primal_solution": w, "dual_solution": q, "metrics": to_dict_of_lists(metrics), "elapsed": toc - tic}
    

def minimize_prospect(
        loss,
        dual_maximization_oracle,
        X,
        y,
        fit_intercept=False,
        X_test=None,
        y_test=None,
        lr=None,
        weight_decay=0.01,
        max_iter=None,
        eval_interval=100,
        uniform=False,
        stopping_tol=1e-3,
        bubble=True,
        seed=1,
        shift_cost=0.05,
        penalty="chi2",
    ):

    tic = time.time()
    
    # initialization
    n, d = X.shape
    if lr is None:
        # define recursive function
        def opt_func(lr, X_, y_):
            return minimize_prospect(
                loss, 
                dual_maximization_oracle,
                X_,
                y_,
                fit_intercept=fit_intercept,
                lr=lr,
                max_iter=min(4 * n, 1600) + 1,
                eval_interval=min(4 * n, 1600),
                weight_decay=weight_decay,
                uniform=uniform,
                bubble=bubble,
            )
        lr = select_learning_rate(opt_func, X, y)

    primal_first_order_oracle = get_primal_first_order_oracle(loss)
    n_class = None
    if loss == "multinomial_cross_entropy":
        n_class = len(np.unique(y))
        d = n_class * d
        primal_first_order_oracle = partial(primal_first_order_oracle, n_class)

    if max_iter is None:
        max_iter = min(100 * n, 200000)

    np.random.seed(seed)
    rng = np.random.RandomState(seed)

    # initialize primal/dual iterates and tables
    w = np.zeros(d, dtype=np.float64) 
    loss_tab, grad_tab = primal_first_order_oracle(w, X, y)
    argsort = np.argsort(loss_tab)
    sorted_losses = loss_tab[argsort]
    inv_perm = np.argsort(argsort)

    grad_tab += compute_regularizer_grad(w, weight_decay, fit_intercept)[None, :]
    dual_val, q = dual_maximization_oracle(loss_tab)
    q_tab = q.copy()
    grad_agg = q_tab @ grad_tab

    w_cert = w.copy()
    q_cert = q.copy()
    primal_grad = grad_tab.T @ q
    linear_gap = q @ (loss_tab - grad_tab @ w)

    # main loop
    metrics = []
    primal_oracle_evals = 0
    iter = 0
    for iter in range(max_iter):

        # evaluation
        if iter % eval_interval == 0:
            metrics.append(
                evaluate(
                    iter, w, X, y, 
                    fit_intercept,
                    primal_first_order_oracle, 
                    dual_maximization_oracle, 
                    weight_decay, 
                    primal_oracle_evals,
                    X_test=X_test,
                    y_test=y_test,
                )
            )
            if iter == 0:
                metrics[-1]["certificate"] = -1
            else:
                if iter == eval_interval:
                    w_cert = w.copy()
                    q_cert = q.copy()
                    loss_, grad_ = primal_first_order_oracle(w, X, y)
                    primal_grad = grad_.T @ q
                    linear_gap = q @ (loss_- grad_ @ w)
                certificate, w_cert, q_cert, primal_grad, linear_gap = compute_certificate(
                    w, 
                    q, 
                    w_cert,
                    q_cert, 
                    primal_grad, 
                    linear_gap, 
                    X, 
                    y, 
                    primal_first_order_oracle, 
                    dual_maximization_oracle, 
                    weight_decay, 
                    fit_intercept,
                    shift_cost=shift_cost,
                    penalty=penalty,
                )
                metrics[-1]["certificate"] = certificate
            if metrics[-1]["gradient_norm"] < stopping_tol * metrics[0]["gradient_norm"]:
                break
            if iter > 0 and metrics[-1]["certificate"] < stopping_tol * metrics[0]["certificate"]:
                break

        # main update
        if uniform:
            i = rng.randint(0, n)
        else:
            q = np.maximum(q, 0)
            q /= q.sum()
            i = np.random.choice(n, size=1, p=q)

        loss, grad = primal_first_order_oracle(w, X[i], y[i])
        grad += compute_regularizer_grad(w, weight_decay, fit_intercept)
        primal_oracle_evals += 1

        if uniform:
            v = n * q[i] * grad - n * q_tab[i] * grad_tab[i] + grad_agg
        else:
            v = (grad - grad_tab[i] + grad_agg)[0]
        
        w -= lr * v

        # update tables and aggregate
        if not uniform:
            i = rng.randint(0, n)
            loss, grad = primal_first_order_oracle(w, X[i], y[i])
            grad += compute_regularizer_grad(w, weight_decay, fit_intercept)

        loss_tab[i] = loss
        if bubble:
            rank = inv_perm[i] # the rank that was just updated
            sorted_losses[rank] = loss
            sorted_losses, argsort, inv_perm = bubble_sort(rank, sorted_losses, argsort, inv_perm)
            dual_val, q = dual_maximization_oracle(loss_tab, sorted_losses=sorted_losses, inv_perm=inv_perm)
        else:
            dual_val, q = dual_maximization_oracle(loss_tab)
        grad_agg += q[i] * grad.reshape(-1) - q_tab[i] * grad_tab[i]
        q_tab[i] = q[i]
        grad_tab[i] = grad

    if iter == max_iter:
        msg = f"Prospect failed to converge within {max_iter} iterations! Gradient norm init: {metrics[0]['gradient_norm']}, final: {metrics[1]['gradient_norm']}"
        warnings.warn(msg, ConvergenceWarning)

    toc = time.time()

    return {"primal_solution": w, "dual_solution": q, "metrics": to_dict_of_lists(metrics), "elapsed": toc - tic}
    

@numba.jit(nopython=True)
def bubble_sort(idx, sort, argsort, inv_perm):
    n = len(sort)

    # Bubble left.
    j = idx
    while j > 0 and sort[j] < sort[j - 1] - 1e-10:
        # Swap elements in sorted vector.
        temp = sort[j]
        sort[j] = sort[j - 1]
        sort[j - 1] = temp

        # Swap elements in rank-to-index vector.
        temp = argsort[j]
        argsort[j] = argsort[j - 1]
        argsort[j - 1] = temp

        # Swap elements in index-to-rank vector.
        inv_perm[argsort[j - 1]] -= 1
        inv_perm[argsort[j]] += 1

        j -= 1

    # Bubble right.
    j = idx
    while j < n - 1 and sort[j] > sort[j + 1] + 1e-10:
        # Swap elements in sorted vector.
        temp = sort[j]
        sort[j] = sort[j + 1]
        sort[j + 1] = temp

        # Swap elements in rank-to-index vector.
        temp = argsort[j]
        argsort[j] = argsort[j + 1]
        argsort[j + 1] = temp

        # Swap elements in index-to-rank vector.
        inv_perm[argsort[j + 1]] += 1
        inv_perm[argsort[j]] -= 1

        j += 1

    return sort, argsort, inv_perm