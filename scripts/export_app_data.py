"""앱·배포용 소형 산출물. 원본 웨이퍼맵(LSWMD)은 재배포하지 않는다 — 파생 결과와 표본만.

artifacts/app_dies.npz : REAL-proxy 대화형 실험용 다이 표본 (웨이퍼 400장의 전 다이, 점수·정답)
"""

from __future__ import annotations

import numpy as np

from stacklens import paths


def main() -> None:
    d = np.load(paths.GOLD / "test_die_scores.npz")
    rng = np.random.default_rng(11)
    wids = rng.choice(np.unique(d["wafer_id"]), 400, replace=False)
    sel = np.isin(d["wafer_id"], wids)
    np.savez_compressed(
        paths.ARTIFACTS / "app_dies.npz",
        y=d["y"][sel].astype(np.int8),
        wafer_id=d["wafer_id"][sel].astype(np.int32),
        B1=d["B1"][sel].astype(np.float16),
        B2=d["B2"][sel].astype(np.float16),
        B4=d["B4"][sel].astype(np.float32),
    )
    size = (paths.ARTIFACTS / "app_dies.npz").stat().st_size
    print(f"다이 {sel.sum():,}개 · {size / 1e6:.1f}MB")
    # passing_die_risk.npz는 전체 통과 다이라 크다 → 앱용 20만 표본으로 대체
    p = np.load(paths.GOLD / "passing_die_risk.npz")
    idx = rng.choice(len(p["risk"]), 200_000, replace=False)
    np.savez_compressed(
        paths.ARTIFACTS / "passing_die_risk_sample.npz", risk=p["risk"][idx].astype(np.float32)
    )
    print(
        f"통과 다이 위험 표본 {(paths.ARTIFACTS / 'passing_die_risk_sample.npz').stat().st_size / 1e6:.1f}MB"
    )


def export_risk_maps() -> None:
    """앱 '다이 위험' 탭용: 패턴이 뚜렷한 test 웨이퍼의 실제 결과와 교차 마스킹 위험 지도."""
    import json

    import duckdb
    import lightgbm as lgb

    from stacklens.pipeline.ingest import decode_map
    from stacklens.risk.features import cross_masked_features

    pattern_model = lgb.Booster(model_file=str(paths.ARTIFACTS / "models" / "pattern_lgbm.txt"))
    risk_model = lgb.Booster(model_file=str(paths.ARTIFACTS / "models" / "risk_lgbm.txt"))
    wids = np.unique(np.load(paths.GOLD / "test_die_scores.npz")["wafer_id"])
    glob = str(paths.SILVER / "wafers" / "part-*.parquet")
    df = (
        duckdb.connect()
        .execute(
            f"select wafer_id, pattern, rows, cols, map_bytes from read_parquet('{glob}') "
            f"where wafer_id in ({','.join(map(str, wids))}) and labeled order by wafer_id"
        )
        .df()
    )
    rng = np.random.default_rng(4)
    out = []
    for pat in ["Edge-Ring", "Loc", "Scratch", "Center", "Edge-Loc", "Donut", "Random", "none"]:
        cand = df[df.pattern == pat]
        for k in rng.choice(len(cand), min(2, len(cand)), replace=False):
            r = cand.iloc[k]
            m = decode_map(r.map_bytes, r.rows, r.cols)
            X, y, xy = cross_masked_features(m, pattern_model)
            risk = risk_model.predict(X)
            out.append(
                {
                    "wafer_id": int(r.wafer_id),
                    "pattern": pat,
                    "shape": list(m.shape),
                    "coords": xy.tolist(),
                    "fail": y.tolist(),
                    "risk": np.round(risk, 4).tolist(),
                }
            )
    (paths.ARTIFACTS / "risk_maps.json").write_text(json.dumps(out))
    print(f"위험 지도 {len(out)}장 → artifacts/risk_maps.json")


if __name__ == "__main__":
    main()
    export_risk_maps()
