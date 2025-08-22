"""Functions for implementing objective function and gradient oracles."""
import numpy as np
import scipy

def get_primal_first_order_oracle(loss_name):
    if loss_name == "squared_error":
        return squared_error_first_order_oracle
    elif loss_name == "binary_cross_entropy":
        return binary_cross_entropy_first_order_oracle
    elif loss_name == "multinomial_cross_entropy":
        return multinomial_cross_entropy_first_order_oracle
    else:
        raise NotImplementedError(f"Loss {loss_name} is not implemented!")
    
def squared_error_first_order_oracle(w, X, y):
    losses = 0.5 * (y - X @ w) ** 2
    if len(X.shape) > 1:
        grads = (X @ w - y)[:, None] * X
    else:
        grads = (X @ w - y) * X
    return losses, grads

def binary_cross_entropy_first_order_oracle(w, X, y):
    n = len(X)
    logits = X @ w
    if len(X.shape) > 1:
        p = scipy.special.expit(logits)
        losses = -np.log(np.stack([1 - p, p], axis=1)[np.arange(n), y])
        grads = (p - y)[:, None] * X
    else:
        p = scipy.special.expit(logits)
        losses = -np.log([1 - p, p][y])
        grads = (p - y) * X

    return losses, grads

def multinomial_cross_entropy_first_order_oracle(n_class, w, X, y):
    n = len(X)
    W = w.reshape(-1, n_class)

    logits = X @ W
    if len(X.shape) > 1:
        log_probs = scipy.special.log_softmax(logits, axis=1)
        losses = -log_probs[np.arange(n), y]

        p = scipy.special.softmax(logits, axis=1)
        p[np.arange(n), y] -= 1
        scores = np.matmul(X[:, :, None], p[:, None, :])
        grads = scores.reshape(n, -1)
    else:
        log_probs = scipy.special.log_softmax(logits)
        losses = -log_probs[y]

        p = scipy.special.softmax(logits)
        p[y] -= 1
        scores = X[:, None] @ p[None, :]
        grads = scores.reshape(-1)

    return losses, grads