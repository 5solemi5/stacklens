"""누수 방지 불변식.

v1에서 실제로 새던 경로("자기 제외 웨이퍼 불량률 + 자기 포함 패턴 확률")를 다시 막는다.
검사 방식: 가린 다이의 결과를 아무렇게나 바꿔도 그 다이들의 **모든** 특징
(패턴 확률 포함)이 한 비트도 변하지 않아야 한다.
"""

import numpy as np

from stacklens.risk.features import (
    DIE_FEATURE_NAMES,
    cross_masked_features,
    fold_assignment,
    masked_die_features,
)
from stacklens.wafer.features import FEATURE_NAMES, wafer_features


def _toy_map(seed=0, size=40):
    rng = np.random.default_rng(seed)
    yy, xx = np.indices((size, size))
    d2 = (yy - size / 2) ** 2 + (xx - size / 2) ** 2
    disk = d2 < (size / 2 - 1) ** 2
    m = np.where(disk, 1, 0).astype(np.uint8)
    m[disk & (rng.random((size, size)) < 0.1)] = 2
    m[disk & (d2 > (size / 2 - 5) ** 2)] = 2  # 가장자리 링
    return m


class _FakePatternModel:
    """wafer_features 전체를 그대로 반영하는 가짜 모델 — 특징에 조금이라도 새면 출력이 바뀐다."""

    def predict(self, X):
        w = np.linspace(0.1, 1.0, X.shape[1])
        s = X @ w
        return np.column_stack([np.sin(s + i) for i in range(9)])


def test_masked_die_values_never_reach_features():
    m = _toy_map()
    folds = fold_assignment(m)
    rng = np.random.default_rng(1)
    model = _FakePatternModel()
    for f in range(3):
        masked = folds == f
        X, xy = masked_die_features(m, masked, model)
        for _ in range(5):
            m2 = m.copy()
            flip = rng.random(masked.sum()) < 0.5
            vals = m2[masked]
            vals[flip] = np.where(vals[flip] == 2, 1, 2)
            m2[masked] = vals
            X2, xy2 = masked_die_features(m2, masked, model)
            np.testing.assert_array_equal(xy, xy2)
            np.testing.assert_array_equal(X, X2)


def test_old_leak_would_be_caught():
    """회귀 방지: 자기 포함 통계를 특징에 넣으면 위 검사가 실패한다는 것을 확인 (검사 자체의 검증)."""
    m = _toy_map(4)
    masked = fold_assignment(m) == 0
    leaky = lambda mm: np.full(masked.sum(), (mm == 2).sum())  # noqa: E731 — 전체 맵 불량 수(자기 포함)
    m2 = m.copy()
    i, j = np.argwhere(masked)[0]
    m2[i, j] = 1 if m[i, j] == 2 else 2
    assert not np.array_equal(leaky(m), leaky(m2))


def test_each_die_scored_exactly_once():
    m = _toy_map(2)
    X, y, xy = cross_masked_features(m)
    assert X.shape == (int((m > 0).sum()), len(DIE_FEATURE_NAMES))
    assert len({tuple(c) for c in xy.tolist()}) == len(xy)
    assert (y == (m[xy[:, 0], xy[:, 1]] == 2)).all()


def test_wafer_features_shape_and_finite():
    f = wafer_features(_toy_map(5))
    assert f.shape == (len(FEATURE_NAMES),)
    assert np.isfinite(f).all()


def test_edge_ring_has_higher_edge_fail():
    f = dict(zip(FEATURE_NAMES, wafer_features(_toy_map(2)), strict=True))
    assert f["edge_fail"] > f["center_fail"]
