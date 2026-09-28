"""M1 · 웨이퍼 패턴 분류 (REAL).

측정하는 것
1. lot 분할 vs 웨이퍼 무작위 분할 — 같은 lot 웨이퍼가 train/test에 섞이면 성능이 얼마나 부풀려지나
2. 클래스 불균형 처리 — 가중 없음 vs balanced
3. 전부 none이라고 답하는 베이스라인 대비
모델은 3개 시드로 돌려 평균±표준편차를 보고한다.

산출물: artifacts/results/pattern.json, artifacts/models/pattern_lgbm.txt(train lot만으로 학습),
        artifacts/wafer_samples.npz(앱 표시용 소량 맵)
"""

from __future__ import annotations

import json
import time

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from stacklens import paths
from stacklens.pipeline.contracts import PATTERNS
from stacklens.pipeline.ingest import decode_map
from stacklens.wafer.features import FEATURE_NAMES, wafer_features

SEEDS = [0, 1, 2]
LGB_PARAMS = dict(
    n_estimators=600,
    learning_rate=0.05,
    num_leaves=63,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    verbose=-1,
)


def load_labeled() -> pd.DataFrame:
    glob = str(paths.SILVER / "wafers" / "part-*.parquet")
    return (
        duckdb.connect()
        .execute(
            f"select wafer_id, lot_name, role, pattern, rows, cols, map_bytes, fail_rate from read_parquet('{glob}') where labeled"
        )
        .df()
    )


def featurize(df: pd.DataFrame) -> np.ndarray:
    return np.vstack(
        [wafer_features(decode_map(b, r, c)) for b, r, c in zip(df.map_bytes, df.rows, df.cols, strict=True)]
    )


def fit_eval(Xtr, ytr, Xva, yva, Xte, yte, seed, class_weight):
    clf = lgb.LGBMClassifier(**LGB_PARAMS, random_state=seed, class_weight=class_weight)
    clf.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(50, verbose=False)])
    pred = clf.predict(Xte)
    return clf, pred, f1_score(yte, pred, average="macro")


def main() -> None:
    t0 = time.time()
    df = load_labeled()
    print(f"라벨 웨이퍼 {len(df):,}장 특징 추출…")
    X = featurize(df)
    y = df.pattern.map({p: i for i, p in enumerate(PATTERNS)}).to_numpy()
    print(f"  {time.time() - t0:.0f}s")

    role = df.role.to_numpy()
    tr, va, te = role == "train", role == "valid", role == "test"
    results = {"n_wafers": int(len(df)), "n_lots": int(df.lot_name.nunique()), "features": FEATURE_NAMES}

    # 베이스라인: 전부 none
    none_idx = PATTERNS.index("none")
    results["baseline_all_none_macro_f1"] = float(
        f1_score(y[te], np.full(te.sum(), none_idx), average="macro")
    )
    results["baseline_all_none_accuracy"] = float((y[te] == none_idx).mean())

    # 1) lot 분할, 가중 여부 비교
    for cw in [None, "balanced"]:
        scores, per_class = [], []
        for s in SEEDS:
            clf, pred, f1 = fit_eval(X[tr], y[tr], X[va], y[va], X[te], y[te], s, cw)
            scores.append(f1)
            per_class.append(f1_score(y[te], pred, average=None, labels=range(len(PATTERNS))))
        key = f"lot_split_{cw or 'unweighted'}"
        results[key] = {
            "macro_f1_mean": float(np.mean(scores)),
            "macro_f1_std": float(np.std(scores)),
            "per_class_f1_mean": dict(
                zip(PATTERNS, np.mean(per_class, axis=0).round(4).tolist(), strict=True)
            ),
        }
        print(f"  {key}: macro-F1 {np.mean(scores):.4f} ± {np.std(scores):.4f}")

    best_cw = max(
        [None, "balanced"], key=lambda c: results[f"lot_split_{c or 'unweighted'}"]["macro_f1_mean"]
    )
    results["chosen_class_weight"] = best_cw or "unweighted"

    # 2) 무작위 분할 (같은 크기, lot 무시) — 누수 부풀림 측정
    idx = np.arange(len(df))
    rnd = []
    for s in SEEDS:
        i_tr, i_rest = train_test_split(idx, train_size=tr.sum(), random_state=s, stratify=y)
        i_va, i_te = train_test_split(i_rest, train_size=va.sum(), random_state=s, stratify=y[i_rest])
        _, _, f1 = fit_eval(X[i_tr], y[i_tr], X[i_va], y[i_va], X[i_te], y[i_te], s, best_cw)
        rnd.append(f1)
    results["random_split"] = {"macro_f1_mean": float(np.mean(rnd)), "macro_f1_std": float(np.std(rnd))}
    print(f"  random_split: macro-F1 {np.mean(rnd):.4f} ± {np.std(rnd):.4f}")

    # 3) 최종 모델(train lot만) — 혼동행렬·오분류 표본·다이 위험 모델용 패턴 확률 제공
    clf, pred, f1 = fit_eval(X[tr], y[tr], X[va], y[va], X[te], y[te], 0, best_cw)
    results["final_test_macro_f1"] = float(f1)
    results["confusion_matrix"] = confusion_matrix(y[te], pred, labels=range(len(PATTERNS))).tolist()
    results["classification_report"] = classification_report(
        y[te], pred, labels=range(len(PATTERNS)), target_names=PATTERNS, output_dict=True, zero_division=0
    )
    imp = pd.Series(clf.booster_.feature_importance("gain"), index=FEATURE_NAMES).sort_values(ascending=False)
    results["top_features_gain"] = imp.head(12).round(1).to_dict()

    (paths.ARTIFACTS / "models").mkdir(parents=True, exist_ok=True)
    clf.booster_.save_model(str(paths.ARTIFACTS / "models" / "pattern_lgbm.txt"))

    # 앱 표시용 표본: 클래스별 정답 6장, 오분류 24장 (64×64 nearest 축소, uint8)
    te_idx = np.nonzero(te)[0]
    rng = np.random.default_rng(0)
    pick, kinds = [], []
    for c in range(len(PATTERNS)):
        ok = te_idx[(y[te] == c) & (pred == c)]
        pick += list(rng.choice(ok, min(6, len(ok)), replace=False))
        kinds += ["correct"] * min(6, len(ok))
    wrong = te_idx[y[te] != pred]
    w = list(rng.choice(wrong, min(24, len(wrong)), replace=False))
    pick += w
    kinds += ["wrong"] * len(w)
    pos = {g: k for k, g in enumerate(te_idx)}
    maps = []
    for g in pick:
        m = decode_map(df.map_bytes.iat[g], df.rows.iat[g], df.cols.iat[g])
        ri = (np.arange(64) * m.shape[0] / 64).astype(int)
        ci = (np.arange(64) * m.shape[1] / 64).astype(int)
        maps.append(m[np.ix_(ri, ci)])
    np.savez_compressed(
        paths.ARTIFACTS / "wafer_samples.npz",
        maps=np.stack(maps),
        true=np.array([PATTERNS[y[g]] for g in pick]),
        pred=np.array([PATTERNS[pred[pos[g]]] for g in pick]),
        kind=np.array(kinds),
        wafer_id=df.wafer_id.to_numpy()[pick],
    )

    results["elapsed_s"] = round(time.time() - t0, 1)
    out = paths.RESULTS / "pattern.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"완료 {results['elapsed_s']}s → {out.relative_to(paths.ROOT)}")


if __name__ == "__main__":
    main()
