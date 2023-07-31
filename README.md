# MIMO achievability bound for the Rayleigh fading channel
Achievability bound is based on the expectation over codewords. This package consists of two modules:
* **cwd_sample** is responsible for single codewords set processing
* **fbl** is responsible for parallel optimization of t-error event probailities
* **parallel** implements a shared memory-based parallelism. Note that passing large matrices in parallel pool is time-consuming. Shared memory is based on [multiprocessing.Array](https://docs.python.org/3/library/multiprocessing.html#multiprocessing.Array). This method requires serialization and deserialization routines to be implemented. Serialization converts conmple-valued matrices to 1D numpy array with real and imaginary parts stored separately.
* **simulation** is responsible for multiple test runnning, saving, and postprocessing

## Usage

See [main.py](main.py) for more details.
To start the simulation, please specify a dictionary with system parameters that must include:
* Blocklength **n**
* The number of information bits **k**
* The number of active users **Ka**
* The number of antennas at the receiver **L**
* The SNR range specified by `snr_db_min`, `snr_db_max`, and `snr_db_step`

## Requirements

* Numerical python [numpy](https://pypi.org/project/numpy/)
* Scientific python [scipy](https://pypi.org/project/scipy/)

## Performance issues

### The choice of optimization method
For the projection-based bound, the function to be optimizaed is a log-determinant of some parametric comination of matrices.
One can use the following formula to obtain gradients with respect to parameters to be optimized:
```math
\frac{d}{dx}\log\left|A\right| = \text{tr}\left(A^{-1}\frac{dA}{dx}\right).
```
In practice, the matrix inversion in the gradient evaluation procedure becomes a bottleneck of the optimization. As a result, we use a gradient-free methods ([Nelder-Mead](https://en.wikipedia.org/wiki/Nelder–Mead_method)).

### Scipy multithreading

`scipy` optimization routines are multithread and there is no way to control the number of threads from python. The best way is to run multiple parallel simulations (specify `n_workers` parameter in the `parallel.py`) and to disable `scipy` multithreading:

```console
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
```
## References
[1] Kirill Andreev, Daria Ustinova, Alexey Frolov, [Unsourced Random Access with the MIMO Receiver: Projection Decoding Analysis](https://arxiv.org/abs/2303.15395)
