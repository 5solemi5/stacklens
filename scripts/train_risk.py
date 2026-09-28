"""M2 · 다이 위험 모델 (REAL) — 가린 다이 예측.

질문: 한 다이의 결과를 가리고 주변만 봤을 때, 그 다이가 불량인지 얼마나 맞힐 수 있나?
      특히 "같은 웨이퍼 안에서" 어느 다이가 더 위험한지를 가를 수 있나? (GDBN의 전제)

데이터 분할 (lot 해시 고정, pipeline/layers.py)
- 패턴 분류기: train lot으로 학습(valid는 early stopping에만 사용)
- 위험 모델:   valid lot으로 학습  → 패턴 분류기가 학습 때 보지 않은 웨이퍼
- 평가:        test lot           → 어떤 모델도 보지 않은 웨이퍼

비교 모델
- B0 웨이퍼 평균: 자기를 뺀 웨이퍼 불량률 (위치 정보 없음)
- B1 GDBN 규칙: 8이웃 중 불량 수 (현업 규칙의 가장 단순한 형태)
- B2 NNR 유사: 5×5 이웃 불량 비율
- B3 LGBM: 공간 특징 전체
- B4 LGBM + 웨이퍼 패턴 확률 (IEEE 2023 GDBN+패턴 연구와 같은 방향)
"""

from __future__ import annotations

import json
import time

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from stacklens import paths
from stacklens.pipeline.contracts import PATTERNS
from stacklens.pipeline.ingest import decode_map
from stacklens.risk.features import (
    DIE_FEATURE_NAMES,
    K_FOLDS,
    cross_masked_features,
    fold_assignment,
    masked_die_features,
)

N_TRAIN_WAFERS = 8000
TRAIN_FOLDS = [0, 1, 2]  # 학습은 웨이퍼당 3/10 폴드만 (비용 절감), 평가는 10폴드 전부
N_TEST_WAFERS = 3000
PATTERN_COLS = [f"p_{p}" for p in PATTERNS]


def sample_wafers(role: str, n: int, seed: int) -> pd.DataFrame:
    glob = str(paths.SILVER / "wafers" / "part-*.parquet")
    df = (
        duckdb.connect()
        .execute(
            f"select wafer_id, lot_name, pattern, rows, cols, map_bytes, n_dies, fail_rate from read_parquet('{glob}') where role = '{role}'"
        )
        .df()
    )
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


def build(df: pd.DataFrame, pattern_model: lgb.Booster, folds_to_use=None):
    """교차 마스킹 특징. 각 다이는 자기 폴드를 가린 관측 맵에서만 특징을 얻는다."""
    Xs, ys, wids, coords, maps = [], [], [], [], []
    for k in range(len(df)):
        m = decode_map(df.map_bytes.iat[k], df.rows.iat[k], df.cols.iat[k])
        X, y, xy = cross_masked_features(m, pattern_model, folds_to_use=folds_to_use)
        Xs.append(X)
        ys.append(y)
        wids.append(np.full(len(y), df.wafer_id.iat[k], dtype=np.int64))
        coords.append(xy)
        maps.append(m)
    return np.vstack(Xs), np.concatenate(ys), np.concatenate(wids), np.vstack(coords), maps


def within_wafer_auc(score, y, wid) -> tuple[float, int]:
    """두 클래스가 모두 있는 웨이퍼마다 AUC를 구해 평균한다 (웨이퍼 간 차이를 제거한 순위 능력)."""
    df = pd.DataFrame({"s": score, "y": y, "w": wid})
    aucs = []
    for _, g in df.groupby("w"):
        if 0 < g.y.sum() < len(g):
            aucs.append(roc_auc_score(g.y, g.s))
    return float(np.mean(aucs)), len(aucs)


def capture_at_top(score, y, wid, frac=0.05) -> float:
    """웨이퍼마다 위험 상위 frac 다이가 그 웨이퍼 불량의 몇 %를 담는가 (평균, 불량이 있는 웨이퍼만)."""
    df = pd.DataFrame({"s": score, "y": y, "w": wid})
    caps = []
    for _, g in df.groupby("w"):
        nf = g.y.sum()
        if nf == 0 or nf == len(g):
            continue
        k = max(1, int(round(len(g) * frac)))
        top = g.nlargest(k, "s")
        caps.append(top.y.sum() / nf)
    return float(np.mean(caps))


def reliability(p, y, bins=10) -> list[dict]:
    q = np.quantile(p, np.linspace(0, 1, bins + 1))
    out = []
    for a, b in zip(q[:-1], q[1:], strict=True):
        sel = (p >= a) & (p <= b)
        if sel.sum():
            out.append(
                {"pred_mean": float(p[sel].mean()), "obs_rate": float(y[sel].mean()), "n": int(sel.sum())}
            )
    return out


def main() -> None:
    t0 = time.time()
    pattern_model = lgb.Booster(model_file=str(paths.ARTIFACTS / "models" / "pattern_lgbm.txt"))
    names = DIE_FEATURE_NAMES + PATTERN_COLS

    print("학습용(valid lot) 웨이퍼 특징…")
    tr = sample_wafers("valid", N_TRAIN_WAFERS, seed=1)
    Xtr, ytr, wtr, _, _ = build(tr, pattern_model, TRAIN_FOLDS)
    print(f"  다이 {len(ytr):,}개 · 불량률 {ytr.mean():.3%} ({time.time() - t0:.0f}s)")
    print("평가용(test lot) 웨이퍼 특징…")
    te = sample_wafers("test", N_TEST_WAFERS, seed=2)
    Xte, yte, wte, cte, maps_te = build(te, pattern_model)
    print(f"  다이 {len(yte):,}개 · 불량률 {yte.mean():.3%} ({time.time() - t0:.0f}s)")

    col = {n: i for i, n in enumerate(names)}
    n_spatial = len(DIE_FEATURE_NAMES)

    # 학습 웨이퍼 중 10%를 웨이퍼 단위로 떼어 early stopping에 쓴다
    uw = np.unique(wtr)
    rng = np.random.default_rng(0)
    es_w = set(rng.choice(uw, len(uw) // 10, replace=False))
    es = np.isin(wtr, list(es_w))

    def fit(cols):
        m = lgb.LGBMClassifier(
            n_estimators=800,
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=200,
            subsample=0.5,
            subsample_freq=1,
            colsample_bytree=0.8,
            verbose=-1,
            random_state=0,
        )
        m.fit(
            Xtr[~es][:, cols],
            ytr[~es],
            eval_set=[(Xtr[es][:, cols], ytr[es])],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        return m

    print("모델 학습…")
    spatial_cols = list(range(n_spatial))
    all_cols = list(range(len(names)))
    m3 = fit(spatial_cols)
    m4 = fit(all_cols)
    print(f"  ({time.time() - t0:.0f}s)")

    scores = {
        "B0_wafer_rate": Xte[:, col["wafer_fail_obs"]],
        "B1_gdbn_n8": Xte[:, col["n8_fail"]],
        "B2_nnr_5x5": Xte[:, col["fail_frac_r2"]],
        "B3_lgbm_spatial": m3.predict_proba(Xte[:, spatial_cols])[:, 1],
        "B4_lgbm_spatial_pattern": m4.predict_proba(Xte[:, all_cols])[:, 1],
    }
    res = {
        "n_train_wafers": N_TRAIN_WAFERS,
        "n_test_wafers": N_TEST_WAFERS,
        "n_train_dies": int(len(ytr)),
        "n_test_dies": int(len(yte)),
        "test_fail_rate": float(yte.mean()),
        "models": {},
    }
    for k, s in scores.items():
        wa, nw = within_wafer_auc(s, yte, wte)
        res["models"][k] = {
            "roc_auc": float(roc_auc_score(yte, s)),
            "pr_auc": float(average_precision_score(yte, s)),
            "within_wafer_auc": wa,
            "capture_top5pct": capture_at_top(s, yte, wte, 0.05),
        }
        print(
            f"  {k:26s} AUC {res['models'][k]['roc_auc']:.4f}  PR {res['models'][k]['pr_auc']:.4f}  "
            f"웨이퍼내AUC {wa:.4f} (n={nw})  상위5%포착 {res['models'][k]['capture_top5pct']:.3f}"
        )
    res["n_wafers_within_auc"] = nw
    p4 = scores["B4_lgbm_spatial_pattern"]
    res["calibration_B4"] = {"brier": float(brier_score_loss(yte, p4)), "bins": reliability(p4, yte)}
    imp = pd.Series(m4.booster_.feature_importance("gain"), index=names).sort_values(ascending=False)
    res["top_features_gain_B4"] = imp.head(12).round(1).to_dict()

    # 누수 점검(런타임): 가린 다이 값을 뒤집어도 그 다이들의 특징·예측이 그대로인지 실제 데이터로 확인
    rng = np.random.default_rng(3)
    max_delta = 0.0
    for k in rng.choice(len(maps_te), 100, replace=False):
        m = maps_te[k]
        masked = fold_assignment(m) == int(rng.integers(0, K_FOLDS))
        X1, _ = masked_die_features(m, masked, pattern_model)
        m2 = m.copy()
        v = m2[masked]
        m2[masked] = np.where(v == 2, 1, 2)
        X2, _ = masked_die_features(m2, masked, pattern_model)
        max_delta = max(
            max_delta, float(np.abs(m4.predict_proba(X1)[:, 1] - m4.predict_proba(X2)[:, 1]).max())
        )
    res["leak_check"] = {"wafers": 100, "max_abs_pred_delta_when_masked_values_flipped": max_delta}
    print(f"  누수 점검: 가린 다이 값을 전부 뒤집었을 때 예측 변화 최대 {max_delta:.2e}")
    assert max_delta < 1e-9, "누수: 가린 다이의 결과가 특징에 들어갔다"

    # 시뮬레이터 입력: test 웨이퍼의 '통과 다이' 위험 점수 (B4)
    passing = yte == 0
    np.savez_compressed(
        paths.GOLD / "passing_die_risk.npz",
        risk=p4[passing].astype(np.float32),
        wafer_id=wte[passing],
        gdbn_n8=Xte[passing, col["n8_fail"]].astype(np.int8),
    )
    # 앱 표시용: 웨이퍼 몇 장의 다이별 위험 지도
    show = []
    for k in range(0, 12):
        sl = wte == te.wafer_id.iat[k]
        show.append(
            {
                "wafer_id": int(te.wafer_id.iat[k]),
                "pattern": te.pattern.iat[k],
                "shape": list(maps_te[k].shape),
                "coords": cte[sl].tolist(),
                "fail": yte[sl].tolist(),
                "risk": np.round(p4[sl], 4).tolist(),
            }
        )
    (paths.ARTIFACTS / "risk_maps.json").write_text(json.dumps(show))
    m4.booster_.save_model(str(paths.ARTIFACTS / "models" / "risk_lgbm.txt"))
    # REAL-proxy 적층 실험용: test 웨이퍼 전 다이의 점수와 정답 (용량이 커서 data/gold에 둔다, 커밋 안 함)
    paths.GOLD.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        paths.GOLD / "test_die_scores.npz",
        y=yte,
        wafer_id=wte,
        B1=scores["B1_gdbn_n8"].astype(np.float32),
        B2=scores["B2_nnr_5x5"].astype(np.float32),
        B3=scores["B3_lgbm_spatial"].astype(np.float32),
        B4=p4.astype(np.float32),
    )

    res["elapsed_s"] = round(time.time() - t0, 1)
    (paths.RESULTS / "risk.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(f"완료 {res['elapsed_s']}s")


if __name__ == "__main__":
    main()
