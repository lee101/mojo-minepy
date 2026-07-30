import math

import minepy
import numpy as np
import pytest

import mojo_minepy


def scores(data_x, data_y, **kwargs):
    ours = mojo_minepy.MINE(**kwargs)
    reference = minepy.MINE(**kwargs)
    ours.compute_score(data_x, data_y)
    reference.compute_score(data_x, data_y)
    return ours, reference


@pytest.mark.parametrize("estimator", ["mic_approx", "mic_e"])
@pytest.mark.parametrize("seed", [0, 7])
def test_characteristic_matrix_random_parity(estimator, seed):
    rng = np.random.RandomState(seed)
    x = rng.uniform(size=180)
    y = np.sin(5.0 * x) + rng.normal(scale=0.15, size=x.size)
    ours, reference = scores(x, y, alpha=0.65, c=8, est=estimator)
    ours_score = ours.get_score()
    reference_score = reference.get_score()
    assert [row.shape for row in ours_score] == [row.shape for row in reference_score]
    for actual, expected in zip(ours_score, reference_score):
        np.testing.assert_allclose(actual, expected, rtol=0, atol=5e-8)


@pytest.mark.parametrize("estimator", ["mic_approx", "mic_e"])
def test_tied_values_parity(estimator):
    x = np.repeat(np.arange(30, dtype=float), 4)
    y = np.round(np.sin(x / 3.0), 1)
    ours, reference = scores(x, y, alpha=0.7, c=5, est=estimator)
    for actual, expected in zip(ours.get_score(), reference.get_score()):
        np.testing.assert_allclose(actual, expected, rtol=0, atol=5e-8)


def test_simd_tail_parity():
    rng = np.random.RandomState(19)
    x = rng.uniform(size=37)
    y = np.cos(7.0 * x) + rng.normal(scale=0.05, size=x.size)
    ours, reference = scores(x, y, alpha=0.8, c=6)
    for actual, expected in zip(ours.get_score(), reference.get_score()):
        np.testing.assert_allclose(actual, expected, rtol=0, atol=5e-8)


@pytest.mark.parametrize("n", [200, 800], ids=["serial", "parallel"])
def test_orientation_parallel_threshold_parity(n):
    rng = np.random.RandomState(23)
    x = rng.uniform(size=n)
    y = np.sin(9.0 * x) + rng.normal(scale=0.1, size=n)
    ours, reference = scores(x, y)
    for actual, expected in zip(ours.get_score(), reference.get_score()):
        np.testing.assert_allclose(actual, expected, rtol=0, atol=5e-8)


def test_workspace_reused_for_same_geometry():
    rng = np.random.RandomState(29)
    mine = mojo_minepy.MINE()
    mine.compute_score(rng.uniform(size=90), rng.uniform(size=90))
    workspace = mine._workspace
    mine.compute_score(rng.uniform(size=90), rng.uniform(size=90))
    assert mine._workspace is workspace


def test_all_statistics_parity():
    rng = np.random.RandomState(3)
    x = np.linspace(-2, 2, 240)
    y = x * x + rng.normal(scale=0.2, size=x.size)
    ours, reference = scores(x, y, alpha=0.7, c=9)
    assert ours.mic() == pytest.approx(reference.mic(), abs=5e-8)
    assert ours.mas() == pytest.approx(reference.mas(), abs=5e-8)
    assert ours.mev() == pytest.approx(reference.mev(), abs=5e-8)
    assert ours.mcn() == pytest.approx(reference.mcn())
    assert ours.mcn(0.2) == pytest.approx(reference.mcn(0.2))
    assert ours.mcn_general() == pytest.approx(reference.mcn_general())
    assert ours.tic() == pytest.approx(reference.tic(), abs=2e-7)
    assert ours.tic(norm=True) == pytest.approx(reference.tic(norm=True), abs=5e-8)
    for power in (-1, 0, 1, 2):
        assert ours.gmic(power) == pytest.approx(reference.gmic(power), abs=5e-8)


@pytest.mark.parametrize(
    "relationship, expected",
    [
        (lambda x: np.zeros_like(x), (0.0, 0.0, 0.0, 2.0, 2.0)),
        (lambda x: x, (1.0, 0.0, 1.0, 2.0, 2.0)),
        (lambda x: np.sin(8 * np.pi * x), (1.0, 0.875, 1.0, 4.0, 4.0)),
        (lambda x: 2 ** (10 * x), (1.0, 0.0, 1.0, 2.0, 2.0)),
    ],
)
def test_upstream_published_vectors(relationship, expected):
    x = np.linspace(0, 1, 1000)
    mine = mojo_minepy.MINE()
    mine.compute_score(x, relationship(x))
    actual = (mine.mic(), mine.mas(), mine.mev(), mine.mcn(), mine.mcn_general())
    np.testing.assert_allclose(actual, expected, rtol=0, atol=6e-4)


def test_pstats_parity_and_condensed_order():
    rng = np.random.RandomState(11)
    data = rng.uniform(size=(5, 120))
    actual_mic, actual_tic = mojo_minepy.pstats(
        data, alpha=9, c=5, est="mic_e"
    )
    expected_mic, expected_tic = minepy.pstats(data, alpha=9, c=5, est="mic_e")
    assert actual_mic.shape == (10,)
    np.testing.assert_allclose(actual_mic, expected_mic, rtol=0, atol=5e-8)
    np.testing.assert_allclose(actual_tic, expected_tic, rtol=0, atol=5e-8)


def test_cstats_parity():
    rng = np.random.RandomState(12)
    x = rng.uniform(size=(3, 110))
    y = rng.uniform(size=(4, 110))
    actual = mojo_minepy.cstats(x, y, alpha=9, c=5, est="mic_e")
    expected = minepy.cstats(x, y, alpha=9, c=5, est="mic_e")
    assert actual[0].shape == (3, 4)
    np.testing.assert_allclose(actual[0], expected[0], rtol=0, atol=5e-8)
    np.testing.assert_allclose(actual[1], expected[1], rtol=0, atol=5e-8)


def test_computed_state_and_recompute():
    mine = mojo_minepy.MINE(alpha=8)
    assert mine.computed() is False
    for method in (mine.mic, mine.mas, mine.mev, mine.mcn, mine.get_score):
        with pytest.raises(ValueError, match="no score computed"):
            method()
    mine.compute_score([0, 1, 2, 3], [0, 1, 4, 9])
    assert mine.computed() is True
    first = mine.mic()
    mine.compute_score([0, 1, 2, 3], [1, 1, 1, 1])
    assert mine.mic() < first


@pytest.mark.parametrize("alpha", [0, -1, 1.5, 3.999])
def test_invalid_alpha(alpha):
    with pytest.raises(ValueError, match="alpha must"):
        mojo_minepy.MINE(alpha=alpha)


def test_invalid_parameters_and_shapes():
    with pytest.raises(ValueError, match="c must"):
        mojo_minepy.MINE(c=0)
    with pytest.raises(KeyError):
        mojo_minepy.MINE(est="unknown")
    mine = mojo_minepy.MINE()
    with pytest.raises(ValueError, match="shape mismatch"):
        mine.compute_score([1, 2], [1, 2, 3])
    with pytest.raises(ValueError, match="one-dimensional"):
        mine.compute_score([[1, 2]], [[1, 2]])
    with pytest.raises(ValueError, match="shape mismatch"):
        mojo_minepy.cstats(np.zeros((2, 4)), np.zeros((2, 5)))
    with pytest.raises(ValueError, match="two-dimensional"):
        mojo_minepy.pstats(np.zeros(5))
    with pytest.raises(TypeError, match="real-valued"):
        mine.compute_score(np.array([1 + 2j, 3 + 4j]), [1, 2])


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"alpha": np.nan}, "finite"),
        ({"alpha": np.inf}, "finite"),
        ({"c": np.inf}, "finite"),
    ],
)
def test_nonfinite_parameters_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        mojo_minepy.MINE(**kwargs)


def test_ffi_rejects_null_pointer_without_dereference():
    from mojo_minepy._lib import lib

    args = [0] * 8 + [15.0, 0] + [0] * 19
    assert lib().mine_compute_matrix(*args) == 1


def test_public_surface():
    assert mojo_minepy.__all__ == ["MINE", "pstats", "cstats"]
    assert isinstance(mojo_minepy.__version__, str)
    assert math.isfinite(float(mojo_minepy.__version__.split(".")[0]))
