"""Functions for implementing the spectral risk measures via the pool adjacent violators (PAV) algorithm using just-in-time compilation."""
import math
import numpy as np
import numba

def make_spectral_risk_measure_oracle(spectrum, penalty="chi2", shift_cost=0.0):
    if penalty=="chi2":
        # losses are scaled by n to account for chi2 penalty being a factor of 2 * n smaller
        shift_cost = 2 * len(spectrum) * shift_cost
    def dual_max_oracle(losses, sorted_losses=None, inv_perm=None):
        return spectral_risk_measure_maximization_oracle(spectrum, shift_cost, penalty, losses, sorted_losses_=sorted_losses, inv_perm=inv_perm)
    return dual_max_oracle

def spectral_risk_measure_maximization_oracle(spectrum, shift_cost, penalty, losses, sorted_losses_=None, inv_perm=None):
    if shift_cost < 1e-16:
        return spectrum[np.argsort(np.argsort(losses))]
    n = len(losses)
    scaled_losses = losses / shift_cost
    if sorted_losses_ is None or inv_perm is None:
        perm = np.argsort(losses)
        sorted_losses = losses[perm] / shift_cost
        inv_perm = np.argsort(perm)
    else:
        sorted_losses = sorted_losses_ / shift_cost

    if penalty == "chi2":
        primal_sol = l2_centered_isotonic_regression(
            sorted_losses, spectrum
        )
    elif penalty == "kl":
        primal_sol = neg_entropy_centered_isotonic_regression(sorted_losses, spectrum)
    else:
        raise NotImplementedError
    primal_sol = primal_sol[inv_perm]
    if penalty == "chi2":
        q = scaled_losses - primal_sol + 1 / n
        dual_val = q @ losses - shift_cost * 0.5 * ((q - 1 / n) ** 2).sum()
    elif penalty == "kl":
        q = np.exp(scaled_losses - primal_sol) / n
        qp = q[q > 0.0]
        dual_val = q @ losses - shift_cost * (qp * np.log(n * qp)).sum()
    else:
        raise NotImplementedError
    return dual_val, q

def make_superquantile_spectrum(n, head_prob):
    """Construct a superquantile spectrum.

    Args:
        head_prob (float): Head probability between 0 and 1, or the proportion of data that is ignored in the superquantile.

    Returns:
        np.ndarray: The spectrum weights.
    """
    assert head_prob >= 0.0 and head_prob <= 1.0
    if head_prob > 1 - 1e-12:
        return np.ones(n, dtype=np.float64) / n
    spectrum = np.zeros(n, dtype=np.float64)
    idx = math.floor(n * head_prob)
    frac = 1 - (n - idx - 1) / (n * (1 - head_prob))
    if frac > 1e-12:
        spectrum[idx] = frac
        spectrum[(idx + 1) :] = 1 / (n * (1 - head_prob))
    else:
        spectrum[idx:] = 1 / (n - idx)
    return spectrum

def make_extremile_spectrum(n, n_draws):
    """Construct an extremile spectrum. The extremile weights represent the average maximum of `n_draws` i.i.d. samples.

    Args:
        n_draws (float): Number of draws, must be at least 1.0 and can be fractional.

    Returns:
        np.ndarray: The spectrum weights.
    """
    assert n_draws >= 1.0
    return (
        (np.arange(n, dtype=np.float64) + 1) ** n_draws
        - np.arange(n, dtype=np.float64) ** n_draws
    ) / (n**n_draws)


def make_esrm_spectrum(n, risk_aversion):
    """Construct exponential spectral risk measure spectrum.

    Args:
        risk_aversion (float): Risk aversion parameter, must be non-negative.

    Returns:
        np.ndarray: The spectrum weights.
    """
    assert risk_aversion >= 0.0
    if risk_aversion <= 1e-12:
        return np.ones(n, dtype=np.float64) / n
    upper = np.exp(risk_aversion * ((np.arange(n, dtype=np.float64) + 1) / n))
    lower = np.exp(risk_aversion * (np.arange(n, dtype=np.float64) / n))
    return math.exp(-risk_aversion) * (upper - lower) / (1 - math.exp(-risk_aversion))

@numba.jit(nopython=True)
def l2_centered_isotonic_regression(losses, spectrum):
    n = len(losses)
    means = [losses[0] + 1 / n - spectrum[0]]
    counts = [1]
    end_points = [0]
    for i in range(1, n):
        means.append(losses[i] + 1 / n - spectrum[i])
        counts.append(1)
        end_points.append(i)
        while len(means) > 1 and means[-2] >= means[-1]:
            prev_mean, prev_count, prev_end_point = (
                means.pop(),
                counts.pop(),
                end_points.pop(),
            )
            means[-1] = (counts[-1] * means[-1] + prev_count * prev_mean) / (
                counts[-1] + prev_count
            )
            counts[-1] = counts[-1] + prev_count
            end_points[-1] = prev_end_point

    # Expand function so numba understands.
    sol = np.zeros((n,))
    i = 0
    for j in range(len(end_points)):
        end_point = end_points[j]
        sol[i : end_point + 1] = means[j]
        i = end_point + 1
    return sol



@numba.jit(nopython=True)
def neg_entropy_centered_isotonic_regression(losses, spectrum):
    n = len(losses)
    logn = np.log(n)
    log_spectrum = np.log(spectrum)

    lse_losses = [losses[0]]
    lse_log_spectrum = [log_spectrum[0]]
    means = [losses[0] - log_spectrum[0] - logn]
    end_points = [0]
    for i in range(1, n):
        means.append(losses[i] - log_spectrum[i] - logn)
        lse_losses.append(losses[i])
        lse_log_spectrum.append(log_spectrum[i])
        end_points.append(i)
        while len(means) > 1 and means[-2] >= means[-1]:
            prev_mean, prev_lse_loss, prev_lse_log_spectrum, prev_end_point = (
                means.pop(),
                lse_losses.pop(),
                lse_log_spectrum.pop(),
                end_points.pop(),
            )
            new_lse_loss = np.log(np.exp(lse_losses[-1]) + np.exp(prev_lse_loss))
            new_lse_log_spectrum = np.log(np.exp(lse_log_spectrum[-1]) + np.exp(prev_lse_log_spectrum))
            means[-1] = new_lse_loss - new_lse_log_spectrum - logn
            lse_losses[-1], lse_log_spectrum[-1] = new_lse_loss, new_lse_log_spectrum
            end_points[-1] = prev_end_point

    # Expand function so numba understands.
    sol = np.zeros((n,))
    i = 0
    for j in range(len(end_points)):
        end_point = end_points[j]
        sol[i : end_point + 1] = means[j]
        i = end_point + 1
    return sol