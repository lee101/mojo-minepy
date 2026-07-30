"""Benchmark mojo-minepy against upstream minepy on identical inputs."""

from __future__ import annotations

import math
import os
import platform
import sys
import time

import minepy
import numpy as np

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"),
)

import mojo_minepy  # noqa: E402


def best_time(function, repeat=3):
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def cpu_name():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as stream:
            for line in stream:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def mine_case(n, *, alpha=0.6, c=15, est="mic_approx", seed=0):
    rng = np.random.RandomState(seed)
    x = rng.uniform(size=n)
    y = np.sin(12 * x) + 0.25 * rng.normal(size=n)
    ours = mojo_minepy.MINE(alpha=alpha, c=c, est=est)
    reference = minepy.MINE(alpha=alpha, c=c, est=est)
    ours.compute_score(x, y)
    reference.compute_score(x, y)
    np.testing.assert_allclose(ours.mic(), reference.mic(), rtol=0, atol=5e-8)
    return (
        lambda: ours.compute_score(x, y),
        lambda: reference.compute_score(x, y),
    )


def pairwise_case():
    rng = np.random.RandomState(5)
    data = rng.uniform(size=(6, 300))
    actual = mojo_minepy.pstats(data, alpha=0.6, c=15, est="mic_e")
    expected = minepy.pstats(data, alpha=0.6, c=15, est="mic_e")
    np.testing.assert_allclose(actual, expected, rtol=0, atol=5e-8)
    return (
        lambda: mojo_minepy.pstats(data, alpha=0.6, c=15, est="mic_e"),
        lambda: minepy.pstats(data, alpha=0.6, c=15, est="mic_e"),
    )


CASES = [
    ("MIC approx, n=1,000", lambda: mine_case(1_000)),
    ("MIC approx, n=2,000", lambda: mine_case(2_000, seed=1)),
    ("MIC_e, n=2,000", lambda: mine_case(2_000, est="mic_e", seed=2)),
    (
        "MIC approx, n=2,000, alpha=0.7",
        lambda: mine_case(2_000, alpha=0.7, c=8, seed=3),
    ),
    ("pstats, 6 x 300, MIC_e", pairwise_case),
]


def main():
    print(f"Machine: {cpu_name()} ({platform.system()} {platform.machine()})")
    print(f"Python {platform.python_version()}, NumPy {np.__version__}, minepy {minepy.__version__}")
    print()
    print("| case | mojo-minepy | minepy 1.2.6 | result |")
    print("| --- | ---: | ---: | ---: |")
    for name, prepare in CASES:
        ours, reference = prepare()
        ours_time = best_time(ours)
        reference_time = best_time(reference)
        if ours_time <= reference_time:
            result = f"{reference_time / ours_time:.2f}x faster"
        else:
            result = f"{ours_time / reference_time:.2f}x slower"
        print(
            f"| {name} | {ours_time * 1e3:.2f} ms | "
            f"{reference_time * 1e3:.2f} ms | {result} |"
        )


if __name__ == "__main__":
    main()
