"""
This module implements all required parallel routines for bound calculation:
 - Shared memory that keeps sampled codewords and corresponding matrices
 - Per-process random number generation instance seeding
"""

import multiprocessing as mp
import numpy as np

from cwd_sample import CodewordSampleProj, CodewordSampleML

from cwd_sample import DIM_REDUCE

# Variables below are per-process instances initialized at init_pool function
# Random number generator instance
G_RNG_INSTANCE = None
# Shared memory buffer
G_SHM_BUF = None


def init_pool(buf_size, n_workers=None):
    """
    Initialize a multiprocessing pool
    """
    # The number of workers. If None, use the cpu_count() value
    if n_workers is None:
        n_workers = mp.cpu_count()

    # Initialize the seed sequence for child processes:
    manager = mp.Manager()
    queue = manager.Queue()
    for rng in get_seed_list(n_workers):
        queue.put(rng)
    global G_SHM_BUF
    # Workers write to different parts of this array. The lock can be disabled
    G_SHM_BUF = mp.Array('d', buf_size, lock=False)
    return mp.Pool(
        processes=mp.cpu_count(),
        initializer=init_worker,
        initargs=(queue, G_SHM_BUF)
    )


def set_rng_seed(seed_value):
    """
    Set the seed value for multiprocessing pool
    """
    global G_RNG_INSTANCE
    G_RNG_INSTANCE = np.random.default_rng(seed_value)


def init_worker(queue, shm_buf):
    """
    Initialize random number generator seed and set shared memory pointer
    """
    set_rng_seed(queue.get())

    global G_SHM_BUF
    G_SHM_BUF = shm_buf


def get_seed_list(n_workers):
    """
    Generate SeedSequence
    """
    seed_sequence = np.random.SeedSequence()
    return seed_sequence.spawn(n_workers)


def get_shm_buf():
    """
    Get shared memory (inside worker)
    """
    return G_SHM_BUF


def get_rng_instance():
    """
    Get the RNG instance (inside worker)
    """
    return G_RNG_INSTANCE


def matrix2vec(mtx):
    """
    Flatten matrix and split save real and imaginary parts as two arrays.
    Required for shared memory routines
    """
    mtx_flat = mtx.reshape(-1)
    return np.hstack([np.real(mtx_flat), np.imag(mtx_flat)])


class CodewordSampleProjSHM(CodewordSampleProj):
    """
    Methods below correspond to shared memory routines
    Note that multiprocessing.Array does not support complex numbers.
    One must keep real and imaginary parts separately.
    """
    def serialize(self):
        """
        Write to buffer
        :return 1D numpy array of type double
        """
        buf = np.hstack([
            matrix2vec(self.prj_dif),
            matrix2vec(self.prj_crp),
            matrix2vec(self.prj_txp)
        ])
        assert(len(buf) == self.buf_size())
        return buf

    def get_dim(self):
        """
        Get projection matrices dimensionality
        """
        if DIM_REDUCE:
            return min(self.k_a + self.t, self.n) - (self.k_a - self.t)
        return self.n

    def buf_size(self):
        """
        Get buffer size required for serialization
        """
        dim = self.get_dim()
        if DIM_REDUCE:
            return 2 * (dim ** 2 + 2 * dim)
        return 6 * dim ** 2

    @staticmethod
    def max_size(n):
        """
        Get maximum per-sample buffer size given a blocklength
        """
        if DIM_REDUCE:
            return 6 * (n ** 2 + 2 * n)
        return 6 * n ** 2

    def deserialize(self, buf):
        """
        Read data from buffer 'buf', which is a 1D np.array (of type double)
        """
        assert(len(buf) == self.buf_size())
        dim = self.get_dim()
        n0 = dim ** 2
        self.prj_dif = np.array(
            buf[:n0] + 1j * buf[n0:(2 * n0)],
            copy=False
        ).reshape(dim, dim)
        if DIM_REDUCE:
            ofset = 2 * n0
            self.prj_crp = np.array(
                buf[ofset:(ofset + dim)] + 1j * buf[(ofset + dim):(ofset + 2 * dim)],
                copy=False
            )
            self.prj_txp = np.array(
                buf[(ofset + 2 * dim):(ofset + 3 * dim)] +
                1j * buf[(ofset + 3 * dim):(ofset + 4 * dim)],
                copy=False
            )
            assert ofset + 4 * dim == len(buf)
        else:
            self.prj_crp = np.array(
                buf[(2 * n0):(3 * n0)] + 1j * buf[(3 * n0):(4 * n0)],
                copy=False
            ).reshape(dim, dim)
            self.prj_txp = np.array(
                buf[(4 * n0):(5 * n0)] + 1j * buf[(5 * n0):(6 * n0)],
                copy=False
            ).reshape(dim, dim)
            assert 6 * n0 == len(buf)


class CodewordSampleMLSHM(CodewordSampleML):
    """
    Shared memory routines (see comments within CodewordSampleProj)
    """

    def serialize(self):
        """
        Write to buffer
        :return 1D numpy array of type double
        """
        buf = np.hstack([
            matrix2vec(self.inv_crtx),
            matrix2vec(self.inv_rxtx),
            [self.tx_logdet, self.rx_logdet, self.cr_logdet, self.inv_crf_eig_prod],
        ])
        assert(len(buf) == self.buf_size())
        return buf

    def get_dim(self):
        """
        Get matrices dimensionality
        """
        if DIM_REDUCE:
            return min(self.k_a + self.t, self.n)
        return self.n

    def buf_size(self):
        """
        Get buffer size required for serialization
        """
        dim = self.get_dim()
        return 4 * dim ** 2 + 4

    @staticmethod
    def max_size(n):
        """
        Get maximum per-sample buffer size given a blocklength
        """
        return 4 * n ** 2 + 4

    def deserialize(self, buf):
        """
        Read data from buffer 'buf', which is a 1D np.array (of type double)
        """
        assert(len(buf) == self.buf_size())
        dim = self.get_dim()
        n0 = dim ** 2
        self.inv_crtx = np.array(
            buf[:n0] + 1j * buf[n0:(2 * n0)],
            copy=False
        ).reshape(dim, dim)
        self.inv_rxtx = np.array(
            buf[(2 * n0):(3 * n0)] + 1j * buf[(3 * n0):(4 * n0)],
            copy=False
        ).reshape(dim, dim)
        self.tx_logdet = buf[4 * n0]
        self.rx_logdet = buf[4 * n0 + 1]
        self.cr_logdet = buf[4 * n0 + 2]
        self.inv_crf_eig_prod = buf[4 * n0 + 3]
        assert 4 * n0 + 4 == len(buf)


def get_buf_size(settings):
    """
    Get the required shared memory buffer size
    """
    n = settings['n']
    n_samples = settings['n_samples']
    if settings['type'] == 'ml':
        return n_samples * CodewordSampleMLSHM.max_size(n)
    return n_samples * CodewordSampleProjSHM.max_size(n)
