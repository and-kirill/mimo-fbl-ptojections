"""
Simulation routines for MIMO FBL evaluation
"""
import time
import pickle
import os
import logging

import numpy as np

from fbl import AchMIMO, evaluate_ebno, interpolate_raw


LOGGER = logging.getLogger(__name__)


class Simulation:
    """
    Run simulations:
      - Handle which t-values to evaluate
      - Save and load already simulated data
      - Perform postprocessing (write text output)
    """
    def __init__(self, ka, **kwargs):
        LOGGER.info('Simulating Ka = %04d', ka)
        self.snr_range = np.arange(
            Simulation.get_parameter('snr_db_min', **kwargs),
            Simulation.get_parameter('snr_db_max', **kwargs),
            Simulation.get_parameter('snr_db_step', **kwargs)
        )
        # Double-check each parameter:
        fbl_names = ['k', 'n', 'L', 'type', 'cwd_type', 'n_samples']
        self.fbl_settings = {}
        for name in fbl_names:
            self.fbl_settings[name] = Simulation.get_parameter(name, **kwargs)
        self.fbl_settings['Ka'] = ka

        self.target_pupe = Simulation.get_parameter('p_e', **kwargs)
        self.pupe_upper = Simulation.get_parameter('p_e_max', **kwargs)
        self.pupe_lower = Simulation.get_parameter('p_e_stop', **kwargs)

        self.ka = ka
        self.t_indices = get_t_indices(self.ka)
        self.data = self.load_data()

    @staticmethod
    def get_parameter(name, **kwargs):
        """
        Input parameters check
        """
        if name not in Simulation.default_settings():
            raise RuntimeError('Unknown parameter name:', name)
        if name not in kwargs:
            val = Simulation.default_settings()[name]
            LOGGER.warning('Parameter %s not specified. Use default value %s', name, val)
            return val
        return kwargs[name]

    @staticmethod
    def default_settings():
        """
        Default parameters values
        """
        return {
            # Simulation parameters
            # SNR range: do not change after the first data saving!
            'snr_db_min':  -25,
            'snr_db_max':    0,
            'snr_db_step': 0.1,
            'p_e_max':    5e-3,    # Stop evaluation when Pe is higher than this threshold
            'p_e_stop':   5e-4,    # Stop simulations when Pe is smaller than this threshold
            'p_e':        1e-3,    # Target probability of error
            # System parameters (see AchMIMO constructor for description):
            'k': 100,
            'L': 32,
            'n': 1000,
            'n_samples': 192,
            'cwd_type': 'AWGN',
            'type': 'projection',
        }

    def load_data(self):
        """
        Load data. The data is a 2D array. The first axis corresponds to the SNR range
        (do not change the SNR range during the simulations), the second corresponds to
        the t values (from 1 to the number of active users)
        """
        filename = self.get_filename()
        LOGGER.debug('Check filename %s', filename)
        if not os.path.isfile(filename):
            LOGGER.debug('File not found. Create an empty dataset')
            return np.full((len(self.snr_range), self.ka), fill_value=np.inf)
        LOGGER.debug('Loading data from file')
        with open(filename, 'rb') as handle:
            data = pickle.load(handle)
            # Backward compatibility
            if isinstance(data, dict):
                return data['pt_values']
            return data

    def save_data(self):
        """
        Save 2D array of probabilities (SNR values and t-values) to file
        """
        filename = self.get_filename()
        with open(filename, 'wb') as handle:
            pickle.dump(self.data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        LOGGER.debug('Data saved.')

    def run(self, pool, stride=5, start_snr=-np.inf, snr_offset=0.6):
        """
        Run simulations
        :param stride: evaluate each stride-th point in the declared SNR range
        :param start_snr: all SNRs below this value will not be evaluated
        :param snr_offset: if optimal point is found, start evaluation with this optimal value
        minus offset (expressed in dB)
        """
        t_range = np.arange(1, self.ka + 1)
        target_snr, _ = self.postproc()
        LOGGER.info('Simulating Ka = %3d. Target SNR = %1.2f dB.', self.ka, target_snr)
        for snr_id, snr_db in enumerate(self.snr_range):
            # Adjust the initial step
            if np.mod(snr_id, stride) != 0:
                continue
            if snr_db < start_snr:
                continue
            if not np.isinf(target_snr) and snr_db < target_snr - snr_offset:
                continue
            pupe = self.error_prob(snr_id)[0]
            for i in self.t_indices:
                if pupe >= self.pupe_upper:
                    LOGGER.debug('Hit threshold. Proceed to the next SNR point')
                    break
                t = t_range[i]
                if not np.isinf(self.data[snr_id, i]):
                    LOGGER.info('Pt(%03d) = %1.4e. Already evaluated', t, self.data[snr_id, i])
                    continue
                t_start = time.time()
                self.data[snr_id, i] = AchMIMO(snr_db, t, **self.fbl_settings).pt(pool)
                pupe += self.data[snr_id, i] * t / self.ka
                LOGGER.info(
                    'Pt(%03d) = %1.4e. Elapsed %1.2f sec.',
                    t, self.data[snr_id, i], time.time() - t_start
                )
                self.save_data()

            if pupe < self.pupe_lower:
                LOGGER.debug('Hit lower threshold. Proceed to the next point')
                return

    def postproc(self):
        """
        Print optimal Eb/N0 values. Enable logging to see the output
        """
        snr_vals = []
        err_vals = []
        LOGGER.info('Postprocessing started')
        for snr_id, snr_db in enumerate(self.snr_range):
            pe, all_done = self.error_prob(snr_id)
            if all_done:
                snr_vals.append(snr_db)
                err_vals.append(pe)

        snr_vals = np.array(snr_vals)
        err_vals = np.array(err_vals)
        snr_target = interpolate_raw(snr_vals, err_vals, self.target_pupe)
        ebno_db = evaluate_ebno(snr_vals, err_vals, self.target_pupe, **self.fbl_settings)
        LOGGER.info('KA = %04d, SNR = %1.3f dB, Eb/N0 = %1.3f dB.', self.ka, snr_target, ebno_db)
        return snr_target, ebno_db

    def get_filename(self):
        """
        Generate filename from settings
        """
        k = self.fbl_settings['k']
        blocklen = self.fbl_settings['n']
        ant_cnt = self.fbl_settings['L']
        bound_type = self.fbl_settings['type']
        cwd_type = self.fbl_settings['cwd_type']
        return f'data/fbl_{bound_type}_{cwd_type}_k{k}_n{blocklen}_L{ant_cnt}_Ka_{self.ka}.pickle'

    def error_prob(self, snr_id):
        """
        Evaluate error probability
        return: per-user probability error and indicator whether all required t-values are evaluated
        """
        t_range = np.arange(1, self.ka + 1)
        idx = 1 - np.isinf(self.data[snr_id, :])
        pe = np.sum(self.data[snr_id, idx == 1] * t_range[idx == 1]) / self.ka

        all_done = np.sum(idx) == self.ka
        if pe >= 1.0:
            all_done = True

        sign = '=' if all_done else '>'
        if np.sum(idx) > 0:
            LOGGER.info(
                'SNR: %1.2f Ka: %04d. Pe %s %1.3e.',
                self.snr_range[snr_id], self.ka, sign, min(1, pe)
            )
        return pe, all_done


def get_t_indices(ka):
    """
    Get the order of t-values to be evaluated
    """
    delta = 0.5  # Can start simulations with small delta and then gradually increase -> 0.5
    return np.vstack([np.arange(ka), np.arange(ka - 1, -1, -1)]).T.reshape(-1)[:np.int32(2 * delta * ka)]
