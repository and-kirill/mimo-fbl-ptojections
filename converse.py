"""
Print and plot MIMO converse bound (Fano-based)
"""

import numpy as np

from scipy.optimize import root_scalar
from scipy.special import digamma as psi

import matplotlib.pyplot as plt


def entropy_bin(x):
    """
    Binary entropy function. All inputs that are outside [0, 1] interval will be cut
    """
    x = np.minimum(1, np.abs(x))
    return -x * np.log2(x) - (1 - x) * np.log2(1 - x)


def evaluate_converse_csir(snr_db, settings):
    """
    Test whether a CSIR-converse condition is satisfied
    """
    t_range = np.arange(settings['Ka']) + 1
    p_linear = 10 ** (snr_db / 10)
    p_e = settings['p_e']

    lhs = (t_range / settings['Ka'] - p_e) * settings['k'] - entropy_bin(p_e)
    rhs = np.minimum(
        settings['L'] * np.log2(1 + p_linear * t_range),
        t_range * np.log2(1 + p_linear * settings['L'])
    ) * settings['n'] / settings['Ka']
    return 1 - 2 * (np.sum(lhs <= rhs) == settings['Ka'])


def evaluate_converse_nocsi(snr_db, settings):
    """
    Test whether a noSIR-converse condition is satisfied
    """
    p_e = settings['p_e']
    lhs = (1 - p_e) * settings['k'] - entropy_bin(p_e)

    p_linear = 10 ** (snr_db / 10)
    p_total = settings['Ka'] * p_linear
    capacity = settings['n'] * min(np.log2(1 + p_total), p_total * np.log2(np.e))

    n_max = max(settings['Ka'], settings['n'])
    sum_idx = np.arange(min(settings['Ka'], settings['n']))
    tmp = np.sum(psi(n_max - sum_idx) * np.log2(np.e) + np.log2(p_linear + 1 / (n_max - sum_idx)))
    rhs = settings['L'] / settings['Ka'] * (capacity - tmp)

    return 1 - 2 * (lhs <= rhs)


def converse_csir(settings):
    """
    Find the Eb/N0 at which CSIR converse condition is satisfied
    """
    snr2ebno = 10 * np.log10(settings['n'] / settings['k'])
    sol = root_scalar(evaluate_converse_csir, args=settings, bracket=[-50, 50])
    return sol.root + snr2ebno


def converse_nocsi(settings):
    """
    Find the Eb/N0 at which no-CSI converse condition is satisfied
    """
    snr2ebno = 10 * np.log10(settings['n'] / settings['k'])
    sol = root_scalar(evaluate_converse_nocsi, args=settings, bracket=[-50, 50])
    return sol.root + snr2ebno


if __name__ == '__main__':
    sys_params = {
        'n':   1000,  # The number of channel uses
        'k':    100,  # The number of information its
        'L':     32,  # Receiver antenna count
        'p_e': 1e-3,  # Target per-user error
    }
    ka_series = np.arange(5, 900, 5)
    eb_nocsi = np.zeros(ka_series.shape)
    eb_csir = np.zeros(ka_series.shape)
    print('Ka NOCIS CSIR')
    for i, k_a in enumerate(ka_series):
        sys_params['Ka'] = k_a
        eb_nocsi[i] = converse_nocsi(sys_params)
        eb_csir[i] = converse_csir(sys_params)
        print(f'{k_a} {eb_nocsi[i]:1.3e} {eb_csir[i]:1.3e}')

    plt.plot(ka_series, eb_nocsi, label='no-CSI')
    plt.plot(ka_series, eb_csir, label='CSIR')
    plt.grid()
    plt.legend()
    plt.show()
