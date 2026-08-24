# mojo-minepy

`mojo-minepy` is a Mojo port of the compute-heavy core of
[minepy](https://minepy.readthedocs.io/), the Maximal Information-based
Nonparametric Exploration library. It calculates the maximal information
coefficient (MIC) and the other statistics derived from minepy's
characteristic matrix, while exposing a familiar NumPy-facing Python API.

The public Python surface of minepy 1.2.6 is covered:

| API | Coverage |
| --- | --- |
| `MINE(alpha=0.6, c=15, est="mic_approx")` | `mic_approx` and `mic_e` |
| `compute_score`, `get_score`, `computed` | Full characteristic/equicharacteristic matrix |
| `mic`, `mas`, `mev`, `mcn`, `mcn_general`, `gmic`, `tic` | Covered |
| `pstats`, `cstats` | Covered, with upstream condensed/matrix layouts |

The legacy minepy command-line program, C/C++ API, MATLAB API, and MICtools
pipeline are not included. The package name is `mojo_minepy`, so existing code
can use `import mojo_minepy as minepy` for the covered Python subset.

## Install

The Pixi environment includes the pinned Mojo compiler and upstream minepy
1.2.6 used by the parity tests.

```bash
pixi install
pixi run build
pixi run test
```

`pixi run build` produces `dist/libmojo-minepy.so`.

## Usage

```python
import numpy as np
from mojo_minepy import MINE

x = np.linspace(0.0, 1.0, 1000)
y = np.sin(8.0 * np.pi * x)

mine = MINE(alpha=0.6, c=15, est="mic_approx")
mine.compute_score(x, y)

print(round(mine.mic(), 3))  # 1.0
print(round(mine.mas(), 3))  # 0.875
```

For pairwise work:

```python
from mojo_minepy import pstats

mic, normalized_tic = pstats(X, alpha=9, c=5, est="mic_e")
```

## Correctness

The test suite compares full matrices and all derived statistics directly
against the real conda-forge build of minepy 1.2.6. It covers both estimators,
random and tied inputs, non-default parameters, `pstats`, `cstats`, error
behavior, and minepy's published constant, linear, sinusoidal, and exponential
test vectors. Full matrices agree within `5e-8`; the small difference comes
from the two toolchains' math-library implementations.

Run the parity, behavior, FFI-guard, SIMD-tail, parallel-threshold, and
workspace-reuse tests with:

```bash
pixi run test
```

## Performance

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz,
Linux x86-64, Python 3.10.20, NumPy 1.26.4, and minepy 1.2.6. Each entry is the
best of three warmed runs in one process.

| case | mojo-minepy | minepy 1.2.6 | result |
| --- | ---: | ---: | ---: |
| MIC approx, n=1,000 | 41.28 ms | 86.19 ms | 2.09x faster |
| MIC approx, n=2,000 | 132.60 ms | 266.16 ms | 2.01x faster |
| MIC_e, n=2,000 | 63.67 ms | 168.52 ms | 2.65x faster |
| MIC approx, n=2,000, alpha=0.7 | 305.45 ms | 519.69 ms | 1.70x faster |
| pstats, 6 x 300, MIC_e | 23.62 ms | 194.42 ms | 8.23x faster |

The benchmark script checks numerical parity before timing and prints a
Markdown table; no stored numbers are used by the script.

No GPU path is included. The hot work is irregular histogram construction,
strided/gather entropy scans, and a dependency-heavy dynamic program; it does
not offer the roughly greater-than-2-flops-per-byte arithmetic intensity needed
to repay device transfers and launch overhead.

## How it works

NumPy performs the initial index sorts. One ctypes call then passes contiguous
`float64` data, `int64` index arrays, output storage, and scratch storage to the
shared Mojo library as integer addresses. The exported Mojo function rebuilds
`UnsafePointer[..., AnyOrigin[mut=True]]` values inside the C ABI boundary; no
Python objects cross it.

The Mojo kernel performs minepy's equipartitioning, clump/superclump
construction, cumulative histograms, entropy calculation, and dynamic program
for the optimal axis partition. Entropy scans use SIMD with scalar remainder
loops, hoisted reciprocals, and gathered cached logarithms of integer counts.
Large `pstats` and `cstats` workloads evaluate independent pairs in a bounded
thread pool; smaller jobs stay serial. The Python layer presents the rectangular
backing storage as minepy's jagged list of arrays and derives MIC, MAS, MEV,
MCN, GMIC, and TIC from it.

All allocations are caller-owned NumPy arrays and same-geometry calls reuse
their workspace, including sorted-value and grid-width buffers. Python keeps
every contiguous, correctly typed buffer alive
for the duration of the synchronous call. The ABI validates non-null addresses
and geometry before constructing pointers and returns a checked status code.
The Mojo library neither retains pointers nor allocates across the FFI
boundary.

## License

MIT. See `LICENSE`.
