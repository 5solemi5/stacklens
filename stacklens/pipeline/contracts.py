"""데이터 품질 계약.

계약은 "통과/실패"만 말하지 않는다. 몇 건을 봤고 몇 건이 어겼는지, 어긴 예시가
무엇인지를 함께 남긴다. severity가 block인 계약을 어긴 행은 silver로 넘어가지 않고,
warn은 표시만 하고 통과시킨다. 어느 쪽이든 리포트에 남는다.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

PATTERNS = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Near-full", "Random", "Scratch", "none"]
SPLITS = ["Training", "Test"]
MAP_VALUES = {0, 1, 2}  # 0 = 웨이퍼 밖, 1 = 양품 다이, 2 = 불량 다이


@dataclass
class ContractResult:
    name: str
    severity: str  # "block" | "warn"
    description: str
    n_checked: int
    n_failed: int
    examples: list = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.n_failed == 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["passed"] = self.passed
        d["fail_rate"] = self.n_failed / self.n_checked if self.n_checked else 0.0
        return d


def map_stats(map_bytes: bytes, rows: int, cols: int) -> tuple[bool, int, int, int]:
    """(값 도메인 준수 여부, 다이 수, 불량 다이 수, 비정상 값 개수)"""
    a = np.frombuffer(map_bytes, dtype=np.uint8)
    bad_vals = int(np.count_nonzero(a > 2))
    return bad_vals == 0, int(np.count_nonzero(a > 0)), int(np.count_nonzero(a == 2)), bad_vals


def check_bronze(df: pd.DataFrame) -> tuple[list[ContractResult], pd.DataFrame]:
    """bronze 한 파티션을 검사한다. 행 단위 플래그 DataFrame도 함께 돌려준다."""
    stats = [map_stats(b, r, c) for b, r, c in zip(df.map_bytes, df.rows, df.cols, strict=True)]
    domain_ok = np.array([s[0] for s in stats])
    n_dies = np.array([s[1] for s in stats])
    n_fail = np.array([s[2] for s in stats])

    flags = pd.DataFrame(
        {
            "wafer_id": df.wafer_id.to_numpy(),
            "domain_ok": domain_ok,
            "shape_ok": (df.rows.to_numpy() > 0) & (df.cols.to_numpy() > 0),
            "n_dies": n_dies,
            "n_fail": n_fail,
            "die_size_match": n_dies == df.die_size.to_numpy(),
            "label_known": df.failure_type_raw.isna() | df.failure_type_raw.isin(PATTERNS),
            "split_known": df.split_raw.isna() | df.split_raw.isin(SPLITS),
            "has_dies": n_dies > 0,
        }
    )

    def res(name, severity, desc, ok_mask, id_col="wafer_id"):
        bad = ~ok_mask
        ex = df.loc[bad, id_col].head(5).tolist()
        return ContractResult(name, severity, desc, int(len(ok_mask)), int(bad.sum()), ex)

    results = [
        res("C1_map_value_domain", "block", "맵 값은 {0,1,2}만 허용", flags.domain_ok.to_numpy()),
        res("C2_map_shape_positive", "block", "맵 행·열 수 > 0", flags.shape_ok.to_numpy()),
        res("C3_has_dies", "block", "다이가 1개 이상 존재", flags.has_dies.to_numpy()),
        res(
            "C4_die_size_matches_map",
            "warn",
            "원본 dieSize 컬럼 = 맵의 비0 셀 수",
            flags.die_size_match.to_numpy(),
        ),
        res(
            "C5_label_vocabulary",
            "block",
            f"패턴 라벨은 {PATTERNS} 중 하나 또는 없음",
            flags.label_known.to_numpy(),
        ),
        res(
            "C6_split_vocabulary",
            "warn",
            "원본 분할 라벨은 Training/Test 또는 없음",
            flags.split_known.to_numpy(),
        ),
    ]
    return results, flags


def check_lot_shape_consistency(df: pd.DataFrame) -> ContractResult:
    """같은 lot 안의 웨이퍼는 같은 제품이어야 하므로 맵 크기가 같아야 한다."""
    g = df.groupby("lot_name")[["rows", "cols"]].nunique()
    bad = (g.rows > 1) | (g.cols > 1)
    return ContractResult(
        "C7_lot_shape_consistency",
        "warn",
        "같은 lot 안에서 맵 크기(rows×cols)가 하나",
        int(len(g)),
        int(bad.sum()),
        g.index[bad].tolist()[:5],
    )


def check_duplicate_keys(df: pd.DataFrame) -> ContractResult:
    dup = df.duplicated(["lot_name", "wafer_index"], keep=False)
    return ContractResult(
        "C8_unique_lot_wafer",
        "warn",
        "(lot_name, wafer_index) 조합이 유일",
        int(len(df)),
        int(dup.sum()),
        df.loc[dup, "wafer_id"].head(5).tolist(),
    )


def merge_results(parts: list[list[ContractResult]]) -> list[ContractResult]:
    """파티션별 결과를 계약 이름 기준으로 합친다."""
    by = {}
    for lst in parts:
        for r in lst:
            if r.name not in by:
                by[r.name] = ContractResult(r.name, r.severity, r.description, 0, 0, [])
            m = by[r.name]
            m.n_checked += r.n_checked
            m.n_failed += r.n_failed
            m.examples = (m.examples + r.examples)[:5]
    return list(by.values())
