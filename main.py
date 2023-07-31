import logging
import os
import sys
import numpy as np
from contextlib import closing

import parallel
from simulation import Simulation


def exports():
    """
    Suppress multithreading in linear algebra libraries
    """
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'


def enable_log(name, level=logging.DEBUG):
    """
    Enable logging with proper formats
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setLevel(level)
    sh.setFormatter(logging.Formatter('%(asctime)s %(name)s-%(levelname)s: %(message)s'))
    logger.addHandler(sh)


if __name__ == '__main__':
    enable_log('simulation', logging.INFO)
    exports()

    settings = Simulation.default_settings()
    # Maximum shared memory buffer size
    buf_size = parallel.get_buf_size(settings)
    print('Creating parallel pool...')
    with closing(parallel.init_pool(buf_size)) as pool:
        print('Done.')
        for u in np.arange(50, 800, 50):
            print(f'Simulating {u} users')
            s = Simulation(u, **settings)
            s.run(pool)
            s.postproc()
