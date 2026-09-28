"""bronze → silver.

silver는 "분석에 써도 되는" 층이다.
- block 계약을 어긴 행은 제외하고, 제외 사유별 건수를 남긴다
- 라벨을 정규화하고 웨이퍼 단위 통계(다이 수·불량 수·불량률)를 붙인다
- **분할은 lot 단위 해시로 고정**한다. 같은 lot의 웨이퍼는 같은 공정 이력을 공유하므로
  웨이퍼 단위로 섞으면 테스트 성능이 부풀려진다(이 부풀림 자체를 wafer/에서 측정한다)
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from stacklens.pipeline import contracts

N_FOLDS = 10


def lot_fold(lot_name: str) -> int:
    """lot 이름 → 0..9. 실행·기기와 무관하게 항상 같은 값."""
    return int(hashlib.md5(lot_name.encode()).hexdigest(), 16) % N_FOLDS


def fold_role(fold: int) -> str:
    """0-5 train, 6-7 valid, 8-9 test. 테스트 lot은 어떤 모델 학습에도 쓰지 않는다."""
    if fold <= 5:
        return "train"
    if fold <= 7:
        return "valid"
    return "test"


def bronze_to_silver(bronze_dir, silver_dir) -> dict:
    silver_dir.mkdir(parents=True, exist_ok=True)
    for old in silver_dir.glob("part-*.parquet"):
        old.unlink()

    all_results, frames_for_lot_checks = [], []
    dropped = {}
    kept = 0
    for i, part in enumerate(sorted(bronze_dir.glob("part-*.parquet"))):
        df = pq.read_table(part).to_pandas()
        results, flags = contracts.check_bronze(df)
        all_results.append(results)
        frames_for_lot_checks.append(df[["wafer_id", "lot_name", "wafer_index", "rows", "cols"]])

        block = flags.domain_ok & flags.shape_ok & flags.has_dies & flags.label_known
        for col in ["domain_ok", "shape_ok", "has_dies", "label_known"]:
            dropped[col] = dropped.get(col, 0) + int((~flags[col]).sum())

        out = df.loc[block.to_numpy()].copy()
        f = flags.loc[block.to_numpy()]
        out["n_dies"] = f.n_dies.to_numpy()
        out["n_fail"] = f.n_fail.to_numpy()
        out["fail_rate"] = out.n_fail / out.n_dies
        out["pattern"] = out.failure_type_raw
        out["labeled"] = out.pattern.notna()
        out["fold"] = out.lot_name.map(lot_fold).astype(np.int8)
        out["role"] = out.fold.map(fold_role)
        out = out.drop(columns=["failure_type_raw"])
        out.to_parquet(silver_dir / f"part-{i:04d}.parquet", compression="zstd", index=False)
        kept += len(out)

    meta = pd.concat(frames_for_lot_checks, ignore_index=True)
    merged = contracts.merge_results(all_results)
    merged += [contracts.check_lot_shape_consistency(meta), contracts.check_duplicate_keys(meta)]
    return {
        "contracts": [r.to_dict() for r in merged],
        "rows_in": int(len(meta)),
        "rows_out": int(kept),
        "dropped_by_reason": dropped,
    }
