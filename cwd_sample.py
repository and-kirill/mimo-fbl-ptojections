"""
This module implements codeword sampling routines for ML and projection-based MIMO
with Rayleigh block fading model achievability bound
"""

from math import fabs
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import chi2
from scipy.stats.mstats import gmean

# Some adjustments may be required if too large block length or too large active user count
MIN_LOG_VAL = -1e6


# Enable or disable dimensionality reduction
DIM_REDUCE = True


def safe_log_val(val):
    """
    Return log_value: take a real part and cut too small values
    """
    return max(MIN_LOG_VAL, np.real(val))


class CodewordSample:
    """
    General inner-loop optimization routines for all bound types
    """
    def log_prob_error(self, alpha, beta):
        """
        Get probability of error given the outer-loop parameters
        alpha and beta fixed
        """
        def fcn(x):
            return self.error_log_prob_point(x[0], x[1], alpha, beta)

        return minimize(fcn, [0, 0], method='Nelder-Mead').fun

    def log_prob_noregion(self):
        """
        Evaluate t-errors event probability without a region (debug-only)
        """
        def fcn(x):
            return self.error_log_prob_point(x[0], 0, 0, 0)
        return minimize(fcn, [0], method='Nelder-Mead').fun

    def log_prob_region(self, alpha, beta):
        """
        Get probability of region given the outer-loop parameters
        alpha and beta fixed
        """
        def fcn(x):
            return self.region_log_prob_point(x, alpha, beta)
        return minimize(fcn, 0, method='Nelder-Mead').fun

    def error_log_prob_point(self, u, v, alpha, beta):
        """
        Wrapper function which takes control over valid inputs
        'u', 'v', 'beta' - non-negative, 0 < 'alpha' < 1.
        """
        return safe_log_val(
            self._error_log_prob_point(fabs(u), fabs(v), min(fabs(alpha), 1), fabs(beta))
        )

    def region_log_prob_point(self, delta, alpha, beta):
        """
        Wrapper function which takes control over valid inputs
        'delta', 'beta' - non-negative, 0 < 'alpha' < 1.
        """
        return safe_log_val(
            self._region_log_prob_point(fabs(delta), min(fabs(alpha), 1), fabs(beta))
        )

    def _error_log_prob_point(self, u, v, alpha, beta):
        """
        Evaluate t-error event for particular set of parameters to be optimized
        Must be implemented within subclass.
        """
        raise NotImplementedError('Implementation within subclass required')

    def _region_log_prob_point(self, delta, alpha, beta):
        """
        Evaluate t-error event for particular set of parameters to be optimized
        Must be implemented within subclass.
        """
        raise NotImplementedError('Implementation within subclass required')


class CodewordSampleProj(CodewordSample):
    """
    Projection-based codewords sample
    """
    def __init__(self, t, **kwargs):
        self.n = kwargs['n']  # The number of channel uses per frame
        self.L = kwargs['L']  # Receiver antenna count
        self.k_a = kwargs['Ka']  # The number of active users
        self.t = t  # The number of errors
        # Complex-valued matrices of size dim_size() X dim_size()
        self.prj_crp = None
        self.prj_txp = None
        self.prj_dif = None

    def generate(self, rng, snr_db, **kwargs):
        """
        Generate codewords, perform dimensionality reduction via a basis change (2t X 2t)
        """
        self.prj_crp, self.prj_txp, self.prj_dif = sample_proj_mtx_reduced(
            rng, snr_db, self.t, kwargs
        )

    def _error_log_prob_point(self, u, v, alpha, beta):
        """
        t-error log probability. Return np.inf if the parameter combination is invalid:
        - minimum matrix eigenvalue condition is not satisfied
        """
        quad_mean = self._quadratic_mean_invlog(u, alpha * v, -v)
        if np.isinf(quad_mean):
            return np.inf
        return self.L * (v * beta * self.n - quad_mean)

    def _region_log_prob_point(self, delta, alpha, beta):
        """
        Region complement log-probability. return np.inf if the parameter combination is invalid
        """
        if DIM_REDUCE:
            quad_mean = self._quadratic_mean_invlog(0, -alpha * delta, delta, diag=True)
        else:
            quad_mean = self._quadratic_mean_invlog(0, -alpha * delta, delta)
        if np.isinf(quad_mean):
            return np.inf
        return self.L * (-delta * beta * self.n - quad_mean)

    def _quadratic_mean_invlog(self, d_w, cp_w, tp_w, diag=False):
        """
        Inverse logarithm of the quadratic form expectation y^H P_0 y
        :param d_w: weight of the RX-TX projection difference
        :param cp_w: weight of the correct codewords complement projection
        :param tp_w: weight of the transmitted codewords complement projection
        :param diag: mark that only diagonal matrices are involved
        """
        if cp_w + tp_w >= 1:
            return -np.inf
        if diag:
            # Reduce the complexity if only diagonal matrices are involved
            vals = 1 - cp_w * self.prj_crp - tp_w * self.prj_txp
            if np.min(vals) <= 0:
                return -np.inf
            val = np.sum(np.log(vals))
        else:
            if DIM_REDUCE:
                prj_w = d_w * self.prj_dif + np.diag(cp_w * self.prj_crp + tp_w * self.prj_txp)
            else:
                prj_w = d_w * self.prj_dif + cp_w * self.prj_crp + tp_w * self.prj_txp
            # Cholesky decomposition: The fastest way to check that matrix is positive definite
            try:
                chol = np.linalg.cholesky(np.eye(prj_w.shape[0]) - prj_w)
            except np.linalg.LinAlgError:
                return -np.inf
            # Determinant of triangular matrix is a product of its diagonal elements
            # Do not forget factor 2 as the matrix square root has been evaluated.
            val = 2 * np.sum(np.log(np.real(np.diag(chol))))

        if not DIM_REDUCE:
            return val

        # Dimensionality reduction:
        with np.errstate(divide='ignore'):
            log_shift = (self.n - min(self.n, self.k_a + self.t)) * np.log(1 - cp_w - tp_w)
        return val + log_shift

    # Gradient for error probability.
    # Note that gradient estimation is numerically expensive.
    # Hence, Nelder-Mead turns out to be the fastest optimization procedure
    # https://www.physicsforums.com/threads/derivative-of-log-determinant-of-a-matrix-w-r-t-a-parameter.354762/
    # def _error_log_prob_grad(self, u, v, alpha):
    #     mtx = u * self.prj_dif + np.diag(alpha * v * self.prj_crp - v * self.prj_txp)
    #     mtx_inv = np.linalg.inv(np.eye(mtx.shape[0]) - mtx)
    #     grad = np.zeros(2,)  # u, v
    #     if DIM_REDUCE:
    #         n_dims = (self.n - min(self.n, self.k_a + self.t))
    #     else:
    #         n_dims = 0
    #     grad[0] = self.L * np.trace(mtx_inv @ self.prj_dif)
    #     grad[1] = self.L * np.trace(mtx_inv @ np.diag(alpha * self.prj_crp - self.prj_txp)) - alpha * n_dims / (1 - alpha * v + v)
    #     return grad

    def plot(self):
        """
        Plot projection matrices. Debug tool
        """
        _, axs = plt.subplots(1, 3)
        axs[0].imshow(np.abs(self.prj_dif))
        axs[0].set_title('Projection difference')
        if DIM_REDUCE:
            axs[1].imshow(np.abs(np.diag(self.prj_txp)))
            axs[2].imshow(np.abs(np.diag(self.prj_crp)))
        else:
            axs[1].imshow(np.abs(self.prj_txp))
            axs[2].imshow(np.abs(self.prj_crp))
        axs[1].set_title('TX complement')
        axs[2].set_title('CR complement')
        plt.show()


class CodewordSampleML(CodewordSample):
    """
    Maximum likelihood codewords sample
    """
    def __init__(self, t, **kwargs):
        self.n = kwargs['n']  # The number of channel uses per frame
        self.L = kwargs['L']  # Receiver antenna count
        self.k_a = kwargs['Ka']  # The number of active users
        self.t = t  # The number of errors

        # These are real numbers
        self.tx_logdet = None
        self.rx_logdet = None
        self.cr_logdet = None
        self.inv_crf_eig_prod = None

        # These are (Ka + t) X (Ka + t) complex matrices
        self.inv_crtx = None
        self.inv_rxtx = None

    def generate(self, rng, snr_db, **kwargs):
        """
        Sample codewords, generate covariance matrices and perform dimensionality reduction
        to Ka + t
        """
        cwd_tx, cwd_rx, cwd_cr, cwd_fa = sample_codewords(rng, snr_db, self.t, kwargs)

        cov_tx = covariance(cwd_tx)
        cov_rx = covariance(cwd_rx)
        cov_cr = covariance(cwd_cr)

        _, self.tx_logdet = np.linalg.slogdet(cov_tx)
        _, self.rx_logdet = np.linalg.slogdet(cov_rx)
        _, self.cr_logdet = np.linalg.slogdet(cov_cr)

        self.inv_rxtx = np.linalg.inv(cov_rx) @ cov_tx
        self.inv_crtx = np.linalg.inv(cov_cr) @ cov_tx

        # Required for ball evaluation
        inv_crf = np.linalg.inv(cov_cr) @ (cwd_fa @ cwd_fa.conj().T)
        d_crf = np.real(np.linalg.eigvals(inv_crf))
        idx = np.argsort(-np.abs(d_crf))
        self.inv_crf_eig_prod = 1 / gmean(d_crf[idx[:self.t]])

    def __log_det_qf(self, u, r, alpha):
        if (1 - alpha) * r < 0:
            return -np.inf
        if DIM_REDUCE:
            dim = min(self.n, self.k_a + self.t)
            # Dimensionality reduction. Do not forget about identity matrices that were cut.
            # This requires a constant to add
            log_b_shift = (self.n - dim) * np.log(1 + (1 - alpha) * r)
        else:
            dim = self.n
            log_b_shift = 0

        try:
            vals = np.linalg.eigvals(
                (1 - u + r) * np.eye(dim)
                + u * self.inv_rxtx
                - r * alpha * self.inv_crtx
            )
        except np.linalg.LinAlgError:
            return -np.inf
        if np.min(vals) <= 0:
            return -np.inf
        val = np.sum(np.log(vals))
        return val + log_b_shift

    def _error_log_prob_point(self, u, v, alpha, beta):
        log_det_qf = self.__log_det_qf(u, v, alpha)
        if np.isinf(log_det_qf):
            return np.inf

        return self.L * (
                (u - v) * self.tx_logdet
                - u * self.rx_logdet
                + v * alpha * self.cr_logdet
                - log_det_qf
                + v * self.n * beta
        )

    def _region_log_prob_point(self, delta, alpha, beta):
        if alpha == 0:
            x1 = np.inf
        else:
            x1 = self.L * self.inv_crf_eig_prod * (
                    self.n * (1 + delta) * (1 - alpha)
                    - alpha * self.cr_logdet
                    + self.tx_logdet
                    - self.n * beta
            ) / alpha

        x2 = self.L * self.n * (1 + delta)
        # The log domain of chi2cdf(x1, ...) + 1 - chi2cdf(x2, ...)
        # 1 - CDF = survival function
        return logsumexp([
            chi2.logcdf(2 * x1, 2 * self.t * self.L),
            chi2.logsf(2 * x2, 2 * self.n * self.L)
        ])


def sample_codewords(rng, snr_db, t, settings):
    """
    General method for codewords sampling. Returns codewords of Ka + t dimensionality
    """
    n = settings['n']
    k_a = settings['Ka']
    cwd_type = settings['cwd_type']

    tx_amp = np.sqrt(10 ** (snr_db / 10))

    assert cwd_type in ['AWGN', 'BPSK', 'PSHELL']
    if cwd_type == 'BPSK':
        cwd_fulldim = (1 - 2 * (rng.random(size=(n, k_a + t)) < 0.5).astype(np.float64)) * tx_amp
    else:
        cwd_fulldim = complex_noise(rng, [n, k_a + t]) * tx_amp
        if cwd_type == 'PSHELL':
            cwd_fulldim = cwd_fulldim / np.linalg.norm(cwd_fulldim, axis=0) * np.sqrt(n) * tx_amp

    if DIM_REDUCE:
        rotation, _, _ = np.linalg.svd(cwd_fulldim)
        cwd = rotation.conj().T @ cwd_fulldim
        # Skip dimensions:
        cwd = cwd[:(k_a + t), :]
    else:
        cwd = cwd_fulldim


    return (
        cwd[:, :k_a],           # Transmitted  codewords
        cwd[:, t:(k_a + t)],    # Received codewords
        cwd[:, t:k_a],          # Correctly received codewords
        cwd[:, k_a:(k_a + t)],  # Falsely detected codewords
    )


def sample_proj_mtx_reduced(rng, snr_db, t, settings):
    """
    Sample projection matrices and switch from Ka + t to 2 x t dimensions
    """
    proj_tx, proj_rx, proj_cr, tx_cov_sqrt = sample_projection_matrices(rng, snr_db, t, settings)

    n = settings['n']
    k_a = settings['Ka']

    if not DIM_REDUCE:
        # Dimensionality reduction disabled. Return n X n matrices
        eye = np.eye(n)
        return (
            tx_cov_sqrt.conj().T @ (eye - proj_cr) @ tx_cov_sqrt,
            tx_cov_sqrt.conj().T @ (eye - proj_tx) @ tx_cov_sqrt,
            tx_cov_sqrt.conj().T @ (proj_rx - proj_tx) @ tx_cov_sqrt
        )

    eye = np.eye(min(k_a + t, n))
    # Jump into subspace orthogonal to correctly received codewords
    proj_crp = tx_cov_sqrt.conj().T @ (eye - proj_cr) @ tx_cov_sqrt
    proj_txp = eye - proj_tx
    rotation = np.linalg.eigh(proj_crp)[1]

    return (
        np.diag(shrink_dim(proj_crp, rotation, n, k_a, t)),
        np.diag(shrink_dim(tx_cov_sqrt.conj().T @ proj_txp @ tx_cov_sqrt, rotation, n, k_a, t)),
        shrink_dim(tx_cov_sqrt.conj().T @ (proj_rx - proj_tx) @ tx_cov_sqrt, rotation, n, k_a, t)
    )


def sample_projection_matrices(rng, snr_db, t, settings):
    """
    Sample projection matrices for projection-based bound
    """
    cwd_tx, cwd_rx, cwd_cr, _ = sample_codewords(rng, snr_db, t, settings)

    proj_tx = proj_matrix(cwd_tx)  # Projection on transmitted codewords
    proj_rx = proj_matrix(cwd_rx)  # Projection on received codewords
    proj_cr = proj_matrix(cwd_cr)  # Projection on correctly received codewords
    tx_cov_sqrt = covariance_sqrt(cwd_tx)  # Transmitted codewords covariance square-root
    return proj_tx, proj_rx, proj_cr, tx_cov_sqrt


def shrink_dim(matrix, rotation, n, k_a, t):
    """
    For projection-based FBL: reduce problem dimensionality from k_a + t to 2 * t
    """
    matrix = rotation.conj().T @ matrix @ rotation
    # Perform dimensionality reduction:
    mask = np.ones(min(k_a + t, n))
    mask[:(k_a - t)] = 0

    # Check that shrinking zero sub-blocks:
    # print('Shrink:', np.linalg.norm(matrix[:, mask == 0]), np.linalg.norm(matrix[mask == 0, :]))
    matrix = matrix[:, mask == 1]
    matrix = matrix[mask == 1, :]
    return matrix


def covariance(cwd):
    """
    Given codewords X, return X * X' + I
    """
    dim = cwd.shape[0]
    return cwd @ cwd.conj().T + np.eye(dim)


def covariance_sqrt(cwd):
    """
    Find the square root of the Hermitean matrix via its eigenvalue decomposition:
    U * sqrt(d). Takes codewords S as inputs, calculates the codewords covariance as I + S * S'
    :return matrix U such that U * U' = I + S * S'
    """
    cov_matrix = covariance(cwd)
    diagonal, rotation = np.linalg.eigh(cov_matrix)
    return rotation @ np.diag(np.sqrt(diagonal))


def proj_matrix(subspace):
    """
    Projection matrix on the subspace spanned by the columns of the input
    """
    return subspace @ np.linalg.pinv(subspace)


def complex_noise(rng, size):
    """
    Generate standard circular complex noise CN(0, 1) of given size
    :param rng: random number generator instance
    :param size: size of the output noise vector
    :return: CN(0, 1) i.i.d. vector
    """
    return (rng.standard_normal(size=size) + 1j * rng.standard_normal(size=size)) / np.sqrt(2)
