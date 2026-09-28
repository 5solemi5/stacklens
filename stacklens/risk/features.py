"""다이 단위 위험 특징 — 교차 마스킹(cross-masking)으로 "가린 다이 예측".

한 다이의 테스트 결과를 **가리고**, 나머지 다이만으로 그 다이가 불량일 확률을 맞힌다.
이 확률이 곧 GDBN이 말하는 "나쁜 동네" 점수다.

### 왜 교차 마스킹인가 (v1의 누수에서 배운 것)
v1은 다이마다 "자기만 뺀" 특징을 만들었다. 이웃 불량 수는 괜찮았지만,
웨이퍼 불량률 (전체−자기)/(n−1) 은 같은 웨이퍼 안에서 **자기가 불량이면 정확히 더 작다.**
그리고 웨이퍼 패턴 확률은 자기를 **포함한** 맵으로 계산됐다. 두 값을 함께 주면 모델이
자기 결과를 복원할 수 있었다(측정: 웨이퍼 평균 기준선의 웨이퍼 내 AUC = 0.000,
자기 값을 뒤집을 때 예측 변화 최대 0.47).

v2 규칙: 웨이퍼의 다이를 K개 폴드로 나누고, 한 폴드를 **통째로 가린 관측 맵**에서
모든 특징(이웃·웨이퍼 통계·패턴 확률)을 계산한다. 가린 다이의 결과는 어떤 특징에도
들어가지 않는다. K번 반복하면 모든 다이가 자기를 보지 않은 점수를 정확히 하나씩 갖는다.
tests/test_risk_features.py 가 "가린 다이 값을 뒤집어도 모든 특징이 같다"를 강제한다.

다이의 **위치**(반경·각도·가장자리)는 테스트 결과가 아니라 물리적 사실이므로 전체 맵에서 계산한다.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from stacklens.wafer.features import polar_coords, wafer_features

SCALES = (1, 2, 3, 5)  # 커널 반지름: 3×3, 5×5, 7×7, 11×11
K_FOLDS = 10

DIE_FEATURE_NAMES = (
    [f"fail_frac_r{k}" for k in SCALES]
    + [f"fail_cnt_r{k}" for k in SCALES]
    + [
        "n8_fail",
        "n4_fail",
        "offwafer_n8",
        "radius",
        "angle_sin",
        "angle_cos",
        "wafer_fail_obs",
        "wafer_n_dies_log",
    ]
)

_K8 = np.ones((3, 3), dtype=np.float32)
_K8[1, 1] = 0
_K4 = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32)


def fold_assignment(m: np.ndarray, k: int = K_FOLDS, seed: int = 0) -> np.ndarray:
    """다이 셀마다 0..k-1 폴드 번호. 다이 없는 셀은 -1. 웨이퍼·시드가 같으면 항상 같다."""
    valid = m > 0
    rng = np.random.default_rng(seed)
    folds = np.full(m.shape, -1, dtype=np.int16)
    n = int(valid.sum())
    folds[valid] = rng.permutation(np.arange(n) % k)
    return folds


def observed_map(m: np.ndarray, masked: np.ndarray) -> np.ndarray:
    """가린 다이를 0(결과 미관측)으로 둔 관측 맵. 패턴 특징 계산용."""
    obs = m.copy()
    obs[masked] = 0
    return obs


def masked_die_features(m: np.ndarray, masked: np.ndarray, pattern_model=None):
    """가린 다이들(masked=True)의 특징. 모든 특징은 가리지 않은 다이의 결과만 쓴다.

    반환: X (가린 다이 수 × 특징 수 [+9 패턴 확률]), 좌표 (i, j)
    """
    valid = m > 0
    observed = valid & ~masked
    fail_obs = ((m == 2) & observed).astype(np.float32)
    obs_f = observed.astype(np.float32)
    ii, jj = np.nonzero(masked & valid)

    feats, fracs, cnts = [], [], []
    for k in SCALES:
        ker = np.ones((2 * k + 1, 2 * k + 1), dtype=np.float32)
        f_sum = ndimage.convolve(fail_obs, ker, mode="constant")[ii, jj]
        v_sum = ndimage.convolve(obs_f, ker, mode="constant")[ii, jj]
        fracs.append(f_sum / np.maximum(v_sum, 1.0))
        cnts.append(f_sum)
    feats += fracs + cnts
    feats.append(ndimage.convolve(fail_obs, _K8, mode="constant")[ii, jj])
    feats.append(ndimage.convolve(fail_obs, _K4, mode="constant")[ii, jj])
    feats.append(8.0 - ndimage.convolve(valid.astype(np.float32), _K8, mode="constant")[ii, jj])

    r, th = polar_coords(valid)  # 위치는 물리적 사실
    feats += [r[ii, jj], np.sin(th[ii, jj]), np.cos(th[ii, jj])]
    n_obs = max(int(observed.sum()), 1)
    feats.append(np.full(len(ii), fail_obs.sum() / n_obs, dtype=np.float32))
    feats.append(np.full(len(ii), np.log(int(valid.sum())), dtype=np.float32))
    X = np.column_stack(feats).astype(np.float32)

    if pattern_model is not None:
        wf = wafer_features(observed_map(m, masked))
        pp = pattern_model.predict(wf[None, :]).astype(np.float32)
        X = np.hstack([X, np.repeat(pp, len(ii), axis=0)])
    return X, np.column_stack([ii, jj]).astype(np.int16)


def cross_masked_features(
    m: np.ndarray, pattern_model=None, k: int = K_FOLDS, folds_to_use=None, seed: int = 0
):
    """K개 폴드를 차례로 가려 다이마다 자기를 보지 않은 특징을 만든다.

    folds_to_use로 일부 폴드만 쓰면 학습 데이터 생성 비용을 줄일 수 있다.
    반환: X, y(불량=1), 좌표
    """
    folds = fold_assignment(m, k, seed)
    Xs, ys, cs = [], [], []
    for f in folds_to_use if folds_to_use is not None else range(k):
        masked = folds == f
        if not masked.any():
            continue
        X, xy = masked_die_features(m, masked, pattern_model)
        Xs.append(X)
        ys.append((m[xy[:, 0], xy[:, 1]] == 2).astype(np.int8))
        cs.append(xy)
    return np.vstack(Xs), np.concatenate(ys), np.vstack(cs)
