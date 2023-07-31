"""
This module implements MIMO achievability bound evaluation for a single codewords sample.
The expectation over codewords is taken within simulation routines by multiple independent tests
"""
import logging

from functools import partial
import numpy as np

from scipy.interpolate import interp1d
from scipy.optimize import minimize, root_scalar, minimize_scalar
from scipy.special import logsumexp
from scipy.stats import chi2

import parallel

LOGGER = logging.getLogger(__name__)


MIN_LOG_VAL = -60  # Do not return log probability values below this threshold
MIN_LOG_PROB = -20  # Stop optimization when a function reaches this value (log probability)


class HitMinFuncVal(Exception):
    """
    Raise this exception when a function to be optimized falls below MIN_LOG_VAL
    """
    def __init__(self, alpha, beta, msg):
        LOGGER.info(
            'Hit minimum function value at %s. alpha = %1.3e, beta = %1.3e',
            msg, alpha, beta
        )
        Exception.__init__(self)


def safe_log_val(val):
    """
    Return log_value: take a real part and cut too small values
    """
    return max(MIN_LOG_VAL, np.real(val))


def evaluate_ebno(snr_vals, pe_vals, pe_target, **kwargs):
    """
    Get the minimum Eb/N0 such that power violation probability (if applicable) is considered
    return: Eb/N0 (dB)
    """
    cwd_type = kwargs['cwd_type']
    snr2ebno = 10 * np.log10(kwargs['n'] / kwargs['k'])
    assert cwd_type in ['AWGN', 'BPSK', 'PSHELL']
    if cwd_type == 'AWGN':
        return snr_get_gaussian(snr_vals, pe_vals, pe_target, **kwargs) + snr2ebno
    return interpolate_raw(np.array(snr_vals), np.array(pe_vals), pe_target) + snr2ebno


def snr_get_gaussian(snr_vals, pe_vals, pe_target, **kwargs):
    """
    Get minimum SNR by taking power constraint violation probability into account
    :param snr_vals: sorted list of SNR values (dB)
    :param pe_vals: error probability list (corresponding to SNR values),
    which obtained with unconstrained codewords sampling.
    The power constraint is taken into account under a gaussian codebook assumption.
    return: Eb/N0 (dB)
    """
    if len(snr_vals) and not np.all(snr_vals[:-1] < snr_vals[1:]):
        raise RuntimeError('SNR values are not in ascending order!')
    if not np.all(pe_vals[:-1] >= pe_vals[1:]):
        LOGGER.warning('Error probabilities are not in descending order. Check simulations!')
    res = minimize_scalar(
        lambda x: snr_get_gaussian_p(x, snr_vals, pe_vals, pe_target, **kwargs),
        bracket=[0.5, 1]
    )
    return res.fun


def snr_get_gaussian_p(p_decay, snr_vals, pe_vals, pe_target, **kwargs):
    """
    Calculate total error probability given power decay value
    """
    p_decay = np.abs(p_decay)
    block_len = kwargs['n']
    snr_offset = -10 * np.log10(p_decay)
    # Probability to violate energy constrain (under Gaussian codebook assumption)
    p0_prob = kwargs['Ka'] * chi2.sf(2 * block_len, 2 * block_len * p_decay)
    # Get shifted SNR range and per-user error list
    return interpolate_raw(
        np.array(snr_vals) + snr_offset,  # Offset the SNR in accordance with power decay factor
        np.array(pe_vals) + p0_prob,  # Add power violation probability
        pe_target
    )


def interpolate_raw(snr_vals, pe_vals, pe_thr):
    """
    Given the SNR values and PUPE values, find an SNR point at which PUPE crosses a threshold
    """
    if not snr_vals.size:
        return np.inf
    if np.max(pe_vals) < pe_thr:
        return snr_vals[0]
    if np.min(pe_vals) > pe_thr:
        return np.inf
    ind = np.argmax(pe_vals < pe_thr)
    f_fit = interp1d(np.log(pe_vals[(ind - 1):]), snr_vals[(ind - 1):], kind='linear')
    return f_fit(np.log(pe_thr))


class AchMIMO:
    """
    This class performs the achievability bound optimization over parameters alpha and beta.
    Instantiates codeword sample class that must implement the following methods:
    - error_log_prob_point
    - region_log_prob_point
    Suffix _point means that all parameters over which the optimization is performed must be passed
    The lack of *_point suffix means that an optimum value is returned

    The FBL achievability has five parameters to optimize:
    - alpha and beta govern outer optimization loop
    - (u,v), and (delta) govern the inner optimization loop (conditioned on alpha and beta above)

    At the first stage, an initial guess is constructed. This is the point where
    error probability and region complement probability are equal.
    At this stage both probabilities are optimized over inner-loop variables.

    This initial guess is used for optimization performed at the second stage, which is a
    joint optimization over five parameters.
    """
    def __init__(self, snr_db, t, **kwargs):
        self.t = t
        self.snr_db = snr_db
        self.settings = kwargs

        required_keys = [
            'k',  # The number of information bits
            'L',  # The number of antennas at the receiver
            'n',  # Block length
            'n_samples',  # Sample set size to calculate expectation over codewords
            'cwd_type',  # Codeword sampling type. Supported "AWGN", "BPSK", "PSHELL"
            # AWGN corresponds to CN(0, P) codewords
            # BPSK corresponds to +/- 1 * sqrt(P) codewords
            # PSHELL corresponds to the power-shell sampling
            'type',  # Bound type. supported "projection" and "ml" values
            'Ka'  # The number of active users
        ]
        if len(kwargs.keys()) != len(required_keys):
            raise RuntimeError('Please check input parameters. Parameter count mismatch')
        for k in required_keys:
            if k not in kwargs:
                raise RuntimeError('Missing parameter:', k)
        self.n_samples = kwargs['n_samples']
        self.bound_type = kwargs['type']

        # Evaluate combinatorial shifts (same codebook case)
        log_ckt = log_cnk(kwargs['Ka'], self.t)
        log_cmt = t * kwargs['k'] * np.log(2) - log_factorial(t)

        self.comb_shift_err = log_cmt + log_ckt
        self.comb_shift_reg = log_ckt

    def pt(self, pool):
        """
        Optimization over alpha conditioned on beta (outer-loop optimization)
        If the function reaches minimum value defined above -> optimization terminates
        """
        self.init_samples(pool)
        try:
            return self.do_evaluate_pt(pool)
        except HitMinFuncVal:
            return np.exp(MIN_LOG_PROB)

    def do_evaluate_pt(self, pool):
        """
        t-error event probability evaluation. Optimization over alpha and beta is performed
        """
        x0 = self.initial_guess(pool)
        if x0 is None:
            return 1.0

        p_error_log, p_region_log = self.evaluate_samples(pool, x0[0], 0)
        if p_error_log > 0 and p_region_log > 0:
            LOGGER.info('Got > 1 value. Optimal parameters: alpha = %1.3e, beta = %1.3e', x0[0], 0)
            return 1.0

        xopt = minimize(lambda x: self.pt_point(pool, x), x0, method='Nelder-Mead')
        alpha, beta = np.abs(xopt.x)
        LOGGER.info('Optimal parameters: alpha = %1.3e, beta = %1.3e', alpha, beta)
        return np.exp(min(0, xopt.fun))

    def pt_noregion(self, pool):
        """
        Evaluate error probability without region optimization
        """
        self.init_samples(pool)
        vals = pool.map(self.evaluate_noregion, range(self.n_samples))
        val = logsumexp(vals) - np.log(self.n_samples)
        return np.exp(min(0, val))

    def initial_guess(self, pool):
        """
        Get an initial guess of parameters to optimize
        """
        alpha_min = 0
        alpha_max = 1
        try:
            sol = root_scalar(
                lambda x: self.prob_diff(pool, x),
                bracket=[alpha_min, alpha_max]
            )
            return np.hstack([sol.root, 0])  # Set beta = 0 for the initial guess
        except ValueError:
            return None

    def prob_diff(self, pool, alpha):
        """
        Log probabilities difference function to find a point where error and region complement
        probabilities are equal.
        """
        p_error_log, p_region_log = self.evaluate_samples(pool, alpha, 0)
        LOGGER.debug(
            'Initial guess at alpha = %1.3e. Error/region log prob %+1.3e/%+1.3e',
            alpha, p_error_log, p_region_log
        )

        # Raise exception if a minimum function value has been reached
        log_p_e = logsumexp([p_error_log, p_region_log])
        if log_p_e < MIN_LOG_PROB:
            raise HitMinFuncVal(alpha, 0, 'initial guess')

        return p_error_log - p_region_log

    def pt_point(self, pool, x):
        """
        Evaluate the probability of t-errors given outer-loop parameters
        """
        alpha, beta = x
        if max(0, min(alpha, 1)) != alpha or beta < 0:
            return np.inf

        p_error_log, p_region_log = self.evaluate_samples(pool, alpha, beta)
        log_p_e = logsumexp([p_error_log, p_region_log])
        LOGGER.debug('Current log p_e: %+1.3e, alpha = %1.3e, beta = %1.3e', log_p_e, alpha, beta)

        # Raise exception if a minimum function value has been reached
        if log_p_e < MIN_LOG_PROB:
            raise HitMinFuncVal(alpha, beta, 'evaluating Pt')

        return log_p_e

    def init_samples(self, pool):
        """
        Initialize codewords samples (parallel operation)
        """
        LOGGER.debug('Initializing samples. t = %d', self.t)
        pool.map(self.init_sample, range(self.n_samples))
        LOGGER.debug('Done...')

    def init_sample(self, indx):
        """
        Initialize a list of codeword samples using an externally created task queue
        """
        sample = get_bound_type(self.settings['type'])(self.t, **self.settings)
        sample.generate(parallel.get_rng_instance(), self.snr_db, **self.settings)
        vec = sample.serialize()
        buf = parallel.get_shm_buf()
        buf[(indx * len(vec)):((indx + 1) * len(vec))] = vec

    def evaluate_samples(self, pool, alpha, beta):
        """
        Evaluate error and region probabilities for all samples.
        Note:
        * For the ML bound, the optimization of region is performed by a common parameter.
        * For the projection-based bound, optimization is performed per-sample.
        """
        if self.settings['type'] == 'projection':
            results = pool.map(
                partial(self.evaluate_sample_proj, alpha=alpha, beta=beta),
                range(self.n_samples)
            )
            results = np.array(results)
            log_p_error = logsumexp(results[:, 0]) - np.log(self.n_samples)
            log_p_region = logsumexp(results[:, 1]) - np.log(self.n_samples)
        else:
            log_p_errors = pool.map(
                partial(self.evaluate_error_ml, alpha=alpha, beta=beta),
                range(self.n_samples)
            )
            log_p_error = logsumexp(log_p_errors) - np.log(self.n_samples)
            log_p_region = self.evaluate_region_ml(pool, alpha, beta)

        # Apply safe_log_val to speed-up calculations
        log_p_error = safe_log_val(log_p_error + self.comb_shift_err)
        log_p_region = safe_log_val(log_p_region + self.comb_shift_reg)
        return log_p_error, log_p_region

    def evaluate_sample_proj(self, indx, alpha, beta):
        """
        Evaluate error and region probabilities simultaneously for the projection-based bound
        """
        sample = self.load_sample(indx)
        return sample.log_prob_error(alpha, beta), sample.log_prob_region(alpha, beta)

    def evaluate_error_ml(self, indx, alpha, beta):
        """
        For the ML-based bound, evaluate error log prob separately
        """
        sample = self.load_sample(indx)
        return sample.log_prob_error(alpha, beta)

    def evaluate_noregion(self, indx):
        """
        Evaluate error probability (per sample) without region optimization given a sample index
        """
        sample = self.load_sample(indx)
        return sample.log_prob_noregion()

    def evaluate_region_ml(self, pool, alpha, beta):
        """
        Evaluate region probability for ML-type bound
        """
        def fcn(x):
            vals = pool.map(
                partial(self.evaluate_region_ml_point, delta=x, alpha=alpha, beta=beta),
                range(self.n_samples)
            )
            return logsumexp(vals)
        solution = minimize(fcn, np.array([0.0]), method='Nelder-Mead')
        return solution.fun - np.log(self.n_samples)

    def evaluate_region_ml_point(self, indx, delta, alpha, beta):
        """
        Per-sample region evaluateion of the ML bound
        """
        sample = self.load_sample(indx)
        return sample.region_log_prob_point(delta, alpha, beta)

    def load_sample(self, indx):
        """
        Load sample from the shared memory (performed by worker)
        """
        buf_np = np.frombuffer(parallel.get_shm_buf())
        sample = get_bound_type(self.settings['type'])(self.t, **self.settings)
        size = sample.buf_size()
        sample.deserialize(buf_np[(indx * size):((indx + 1) * size)])
        return sample


def get_bound_type(bound_type_str):
    """
    Get bound type from string constant. Valid inputs: 'ml', 'projection'
    """
    # Instantiate codeword samples extensions that support serialization (shared memory functional)
    if bound_type_str == 'ml':
        return parallel.CodewordSampleMLSHM
    if bound_type_str == 'projection':
        return parallel.CodewordSampleProjSHM
    raise RuntimeError(f'Unknown bound type: {bound_type_str}')


def log_cnk(n, k):
    """
    The logarithm of binomial coefficient. This method does not check that inputs are integer
    :return log(C_n^k).
    """
    return log_factorial(n) - log_factorial(k) - log_factorial(n - k)


def log_factorial(n):
    """
    Logarithm of the factorial function
    :return log(n!)
    """
    return np.sum(np.log(np.arange(n) + 1))
