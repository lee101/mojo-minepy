"""Python API compatible with minepy's MINE, pstats, and cstats."""

from __future__ import annotations

import math

import numpy as np

from ._lib import addr, lib

_ESTIMATORS = {"mic_approx": 0, "mic_e": 1}


def _parameters(alpha: float, c: float, est: str) -> tuple[float, float, int]:
    alpha = float(alpha)
    c = float(c)
    estimator = _ESTIMATORS[est]
    if not math.isfinite(alpha):
        raise ValueError("alpha must be finite")
    if not math.isfinite(c):
        raise ValueError("c must be finite")
    if ((alpha <= 0.0 or alpha > 1.0) and alpha < 4.0):
        raise ValueError("alpha must be in (0.0, 1.0] or >= 4.0")
    if c <= 0.0:
        raise ValueError("c must be > 0.0")
    return alpha, c, estimator


class MINE:
    """Maximal Information-based Nonparametric Exploration."""

    def __init__(self, alpha=0.6, c=15, est="mic_approx"):
        self.alpha, self.c, self.estimator = _parameters(alpha, c, est)
        self.est = est
        self._matrix: np.ndarray | None = None
        self._widths: np.ndarray | None = None
        self._workspace: tuple[tuple[int, ...], tuple[np.ndarray, ...]] | None = None

    def compute_score(self, x, y):
        if np.iscomplexobj(x) or np.iscomplexobj(y):
            raise TypeError("x and y must be real-valued")
        xa = np.ascontiguousarray(x, dtype=np.float64)
        ya = np.ascontiguousarray(y, dtype=np.float64)
        if xa.ndim != 1 or ya.ndim != 1:
            raise ValueError("x and y must be one-dimensional")
        if xa.shape[0] != ya.shape[0]:
            raise ValueError("x, y: shape mismatch")
        n = xa.shape[0]
        if n < 2:
            raise ValueError("x and y must contain at least two samples")

        bound = max(n**self.alpha, 4.0) if self.alpha <= 1.0 else min(self.alpha, n)
        rows = max(math.floor(bound / 2.0), 2) - 1
        widths = np.array(
            [math.floor(bound / (i + 2.0)) - 1 for i in range(rows)],
            dtype=np.int64,
        )
        cols = int(widths[0])

        ix = np.ascontiguousarray(np.argsort(xa, kind="quicksort"), dtype=np.int64)
        iy = np.ascontiguousarray(np.argsort(ya, kind="quicksort"), dtype=np.int64)
        xx = np.ascontiguousarray(xa[ix])
        yy = np.ascontiguousarray(ya[iy])

        requested_limit = self.c * (cols + 1)
        if requested_limit > np.iinfo(np.int64).max:
            raise OverflowError("c is too large for the kernel")
        max_limit = max(int(requested_limit), 1)
        pstride = min(n, max_limit)
        xstride = cols + 2
        qstride = rows + 1

        geometry = (n, rows, cols, pstride, xstride, qstride)
        if self._workspace is None or self._workspace[0] != geometry:
            integer_log = np.empty(n + 1, dtype=np.float64)
            integer_log[0] = 0.0
            np.log(np.arange(1, n + 1, dtype=np.float64), out=integer_log[1:])
            self._workspace = (
                geometry,
                (
                    np.empty((rows, cols), dtype=np.float64),
                    np.empty((rows, cols), dtype=np.float64),
                    np.empty(n, dtype=np.int64),
                    np.empty(n, dtype=np.int64),
                    np.empty(n, dtype=np.int64),
                    np.empty(pstride, dtype=np.int64),
                    integer_log,
                    np.empty(qstride * pstride, dtype=np.int64),
                    np.empty((pstride + 1) * xstride, dtype=np.float64),
                    np.empty((pstride + 1) ** 2, dtype=np.float64),
                    np.empty(n, dtype=np.int64),
                    np.empty(n, dtype=np.int64),
                    np.empty(n, dtype=np.int64),
                    np.empty(pstride, dtype=np.int64),
                    np.empty(qstride * pstride, dtype=np.int64),
                    np.empty((pstride + 1) * xstride, dtype=np.float64),
                    np.empty((pstride + 1) ** 2, dtype=np.float64),
                ),
            )
        (
            matrix,
            temporary,
            qtmp,
            qmap,
            pmap,
            counts,
            counts_log,
            cumulative,
            information,
            interval_entropy,
            qtmp2,
            qmap2,
            pmap2,
            counts2,
            cumulative2,
            information2,
            interval_entropy2,
        ) = self._workspace[1]

        status = lib().mine_compute_matrix(
            addr(xx),
            addr(yy),
            addr(ix),
            addr(iy),
            n,
            rows,
            cols,
            addr(widths),
            self.c,
            self.estimator,
            addr(matrix),
            addr(temporary),
            addr(qtmp),
            addr(qmap),
            addr(pmap),
            addr(counts),
            addr(counts_log),
            addr(cumulative),
            addr(information),
            addr(interval_entropy),
            pstride,
            xstride,
            addr(qtmp2),
            addr(qmap2),
            addr(pmap2),
            addr(counts2),
            addr(cumulative2),
            addr(information2),
            addr(interval_entropy2),
        )
        if status != 0:
            raise RuntimeError(f"Mojo kernel rejected its buffers or geometry ({status})")
        self._matrix = matrix
        self._widths = widths

    def _score(self) -> tuple[np.ndarray, np.ndarray]:
        if self._matrix is None or self._widths is None:
            raise ValueError("no score computed")
        return self._matrix, self._widths

    def computed(self):
        return self._matrix is not None

    def get_score(self):
        matrix, widths = self._score()
        return [matrix[i, :width].copy() for i, width in enumerate(widths)]

    def mic(self):
        matrix, widths = self._score()
        return max(
            float(np.max(matrix[i, :width])) for i, width in enumerate(widths)
        )

    def mas(self):
        matrix, widths = self._score()
        value = 0.0
        for i, width in enumerate(widths):
            for j in range(width):
                value = max(value, abs(float(matrix[i, j] - matrix[j, i])))
        return value

    def mev(self):
        matrix, widths = self._score()
        value = 0.0
        for i, width in enumerate(widths):
            if i == 0:
                value = max(value, float(np.max(matrix[i, :width])))
            else:
                value = max(value, float(matrix[i, 0]))
        return value

    def mcn(self, eps=0):
        matrix, widths = self._score()
        mic = self.mic()
        value = math.inf
        for i, width in enumerate(widths):
            for j in range(width):
                if matrix[i, j] + 0.0001 >= (1.0 - eps) * mic:
                    value = min(value, math.log2((i + 2) * (j + 2)))
        return value

    def mcn_general(self):
        matrix, widths = self._score()
        mic = self.mic()
        value = math.inf
        for i, width in enumerate(widths):
            for j in range(width):
                if matrix[i, j] + 0.0001 >= mic * mic:
                    value = min(value, math.log2((i + 2) * (j + 2)))
        return value

    def tic(self, norm=False):
        matrix, widths = self._score()
        total = sum(float(np.sum(matrix[i, :width])) for i, width in enumerate(widths))
        if norm:
            total /= int(np.sum(widths))
        return total

    def gmic(self, p=-1):
        matrix, widths = self._score()
        characteristic = []
        for i, width in enumerate(widths):
            for j in range(width):
                bound = (i + 2) * (j + 2)
                sub_rows = max(math.floor(bound / 2.0), 2) - 1
                best = 0.0
                for row in range(sub_rows):
                    sub_width = math.floor(bound / (row + 2.0)) - 1
                    best = max(best, float(np.max(matrix[row, :sub_width])))
                characteristic.append(best)
        values = np.asarray(characteristic)
        if p == 0:
            return float(np.prod(values) ** values.size)
        with np.errstate(divide="ignore", invalid="ignore"):
            return float(np.mean(values ** p) ** (1.0 / p))


def pstats(X, alpha=0.6, c=15, est="mic_approx"):
    data = np.ascontiguousarray(X, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError("X must be two-dimensional")
    count = data.shape[0] * (data.shape[0] - 1) // 2
    mic = np.empty(count, dtype=np.float64)
    tic = np.empty(count, dtype=np.float64)
    mine = MINE(alpha=alpha, c=c, est=est)
    k = 0
    for i in range(data.shape[0] - 1):
        for j in range(i + 1, data.shape[0]):
            mine.compute_score(data[i], data[j])
            mic[k] = mine.mic()
            tic[k] = mine.tic(norm=True)
            k += 1
    return mic, tic


def cstats(X, Y, alpha=0.6, c=15, est="mic_approx"):
    xdata = np.ascontiguousarray(X, dtype=np.float64)
    ydata = np.ascontiguousarray(Y, dtype=np.float64)
    if xdata.ndim != 2 or ydata.ndim != 2:
        raise ValueError("X and Y must be two-dimensional")
    if xdata.shape[1] != ydata.shape[1]:
        raise ValueError("X, Y: shape mismatch")
    mic = np.empty((xdata.shape[0], ydata.shape[0]), dtype=np.float64)
    tic = np.empty_like(mic)
    mine = MINE(alpha=alpha, c=c, est=est)
    for i in range(xdata.shape[0]):
        for j in range(ydata.shape[0]):
            mine.compute_score(xdata[i], ydata[j])
            mic[i, j] = mine.mic()
            tic[i, j] = mine.tic(norm=True)
    return mic, tic
