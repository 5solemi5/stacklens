"""웨이퍼맵 → 수작업 특징.

CNN 전에 해석 가능한 특징부터 만든다. 각 특징은 현업 판독 기준과 대응한다.
- 반경 링별 불량률: Center(중앙) / Edge-Ring(가장자리 링) 구분
- 섹터별 불량률의 편차·최대: Edge-Loc·Loc처럼 한쪽에 몰린 패턴
- 최대 불량 군집의 크기·선형성: Scratch(가늘고 긴 선) / Loc(덩어리)
- 전체 불량률: Near-full / Random

좌표는 다이가 있는 셀의 중심을 원점으로, 가장 먼 다이까지 거리를 1로 정규화한다.
맵 크기가 웨이퍼마다 달라도 같은 특징 공간에 놓기 위해서다.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

N_RINGS = 5
N_SECTORS = 8
_STRUCT8 = np.ones((3, 3), dtype=bool)

FEATURE_NAMES = (
    ["fail_rate", "n_dies_log"]
    + [f"ring{i}_fail" for i in range(N_RINGS)]
    + [f"sector{i}_fail" for i in range(N_SECTORS)]
    + [
        "sector_std",
        "sector_max_ratio",
        "edge_fail",
        "center_fail",
        "edge_center_ratio",
        "n_components",
        "largest_comp_frac",
        "largest_comp_linearity",
        "largest_comp_elong",
        "largest_comp_radius",
        "top3_comp_frac",
        "fail_radius_mean",
        "fail_radius_std",
    ]
)


def polar_coords(valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """다이 셀의 정규화 반경 r(0~1)과 각도 θ(-π~π). 다이가 없는 셀은 nan."""
    ii, jj = np.nonzero(valid)
    ci, cj = ii.mean(), jj.mean()
    di, dj = np.indices(valid.shape)
    di = di - ci
    dj = dj - cj
    rmax = np.sqrt(((ii - ci) ** 2 + (jj - cj) ** 2).max()) or 1.0
    r = np.sqrt(di**2 + dj**2) / rmax
    th = np.arctan2(di, dj)
    r = np.where(valid, r, np.nan)
    th = np.where(valid, th, np.nan)
    return r, th


def _safe_rate(fail_mask: np.ndarray, sel: np.ndarray) -> float:
    n = sel.sum()
    return float(fail_mask[sel].mean()) if n else 0.0


def wafer_features(m: np.ndarray) -> np.ndarray:
    valid = m > 0
    fail = m == 2
    n_dies = int(valid.sum())
    r, th = polar_coords(valid)
    fr = float(fail.sum() / n_dies)

    ring_edges = np.linspace(0, 1.0000001, N_RINGS + 1)
    rings = [
        _safe_rate(fail, valid & (r >= a) & (r < b))
        for a, b in zip(ring_edges[:-1], ring_edges[1:], strict=True)
    ]
    sec = np.floor((np.nan_to_num(th, nan=0.0) + np.pi) / (2 * np.pi) * N_SECTORS).astype(int)
    sec = np.clip(sec, 0, N_SECTORS - 1)  # 다이 없는 셀은 valid 마스크로 걸러진다
    sectors = [_safe_rate(fail, valid & (sec == s)) for s in range(N_SECTORS)]
    sectors_a = np.array(sectors)
    edge = _safe_rate(fail, valid & (r >= 0.8))
    center = _safe_rate(fail, valid & (r <= 0.35))

    lab, ncomp = ndimage.label(fail, structure=_STRUCT8)
    if ncomp:
        sizes = np.bincount(lab.ravel())[1:]
        big = int(np.argmax(sizes)) + 1
        largest = sizes.max() / n_dies
        top3 = np.sort(sizes)[-3:].sum() / n_dies
        pi, pj = np.nonzero(lab == big)
        if len(pi) >= 3:
            cov = np.cov(np.vstack([pi, pj]))
            ev = np.sort(np.linalg.eigvalsh(cov))[::-1]
            linearity = float(ev[0] / (ev.sum() + 1e-9))
            elong = float(np.sqrt(ev[0] / (ev[1] + 1e-3)))
        else:
            linearity, elong = 0.5, 1.0
        comp_r = float(np.nanmean(r[lab == big]))
        fail_r = r[fail]
        fr_mean, fr_std = float(np.nanmean(fail_r)), float(np.nanstd(fail_r))
    else:
        largest = top3 = linearity = comp_r = fr_mean = fr_std = 0.0
        elong = 1.0

    return np.array(
        [fr, np.log(n_dies)]
        + rings
        + sectors
        + [
            float(sectors_a.std()),
            float(sectors_a.max() / (sectors_a.mean() + 1e-6)),
            edge,
            center,
            float(edge / (center + 1e-3)),
            float(np.log1p(ncomp)),
            float(largest),
            linearity,
            elong,
            comp_r,
            float(top3),
            fr_mean,
            fr_std,
        ],
        dtype=np.float32,
    )
