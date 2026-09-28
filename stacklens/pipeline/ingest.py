"""LSWMD.pkl(WM-811K 원본) → bronze Parquet.

원본은 2015년 pandas(0.x)로 저장된 pickle이라 현재 pandas가 모르는 모듈 경로
(`pandas.indexes`, `pandas.core.indexes.numeric` 등)를 참조한다. 역직렬화 시점에만
모듈 별칭을 끼워 넣어 읽는다. 원본 파일은 수정하지 않는다.

bronze는 "원본을 손대지 않고 형식만 바꾼" 층이다. 라벨 정규화나 필터링은 silver에서 한다.
"""

from __future__ import annotations

import pickle
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


class _LegacyUnpickler(pickle.Unpickler):
    """구버전 pandas 모듈 경로를 현재 경로로 바꿔 찾는다."""

    _RENAMES = {
        "pandas.indexes": "pandas.core.indexes",
        "pandas.indexes.base": "pandas.core.indexes.base",
        "pandas.indexes.range": "pandas.core.indexes.range",
        "pandas.indexes.numeric": "pandas.core.indexes.base",
        "pandas.core.indexes.numeric": "pandas.core.indexes.base",
        "pandas.indexes.multi": "pandas.core.indexes.multi",
        "pandas.core.base": "pandas.core.base",
    }
    _CLASS_RENAMES = {
        ("pandas.core.indexes.base", "Int64Index"): ("pandas.core.indexes.base", "Index"),
        ("pandas.core.indexes.base", "Float64Index"): ("pandas.core.indexes.base", "Index"),
        ("pandas.core.indexes.base", "UInt64Index"): ("pandas.core.indexes.base", "Index"),
    }

    def find_class(self, module: str, name: str):
        module = self._RENAMES.get(module, module)
        module, name = self._CLASS_RENAMES.get((module, name), (module, name))
        return super().find_class(module, name)


def load_lswmd(path: Path) -> pd.DataFrame:
    """원본 pickle을 DataFrame으로 읽는다. 약 2GB, 메모리 4~6GB 사용."""
    # 일부 구버전 pickle은 모듈 import 자체를 시도하므로 빈 모듈도 등록해 둔다
    sys.modules.setdefault("pandas.indexes", types.ModuleType("pandas.indexes"))
    with open(path, "rb") as f:
        df = _LegacyUnpickler(f, encoding="latin1").load()
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"예상과 다른 객체: {type(df)}")
    return df


def _unwrap(v) -> str | None:
    """failureType / trianTestLabel 은 [['none']] 같은 중첩 배열이거나 빈 배열이다."""
    arr = np.asarray(v, dtype=object).ravel()
    if arr.size == 0:
        return None
    s = str(arr[0]).strip()
    return s or None


def to_bronze(df: pd.DataFrame, out_dir: Path, chunk: int = 50_000) -> dict:
    """웨이퍼 1장 = 1행. 맵은 uint8 바이트 + (rows, cols)로 저장한다."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("part-*.parquet"):
        old.unlink()  # 멱등: 다시 돌리면 같은 결과

    schema = pa.schema(
        [
            ("wafer_id", pa.int64()),
            ("lot_name", pa.string()),
            ("wafer_index", pa.float64()),
            ("die_size", pa.float64()),
            ("rows", pa.int32()),
            ("cols", pa.int32()),
            ("split_raw", pa.string()),
            ("failure_type_raw", pa.string()),
            ("map_bytes", pa.binary()),
        ]
    )
    # 원본 컬럼명에는 오타(trianTestLabel)가 있다. 그대로 받는다.
    split_col = "trianTestLabel" if "trianTestLabel" in df.columns else "trainTestLabel"
    n = len(df)
    parts = 0
    for start in range(0, n, chunk):
        sub = df.iloc[start : start + chunk]
        maps = [np.asarray(m, dtype=np.uint8) for m in sub["waferMap"]]
        table = pa.table(
            {
                "wafer_id": np.arange(start, start + len(sub), dtype=np.int64),
                "lot_name": sub["lotName"].astype(str).to_numpy(),
                "wafer_index": sub["waferIndex"].astype(float).to_numpy(),
                "die_size": sub["dieSize"].astype(float).to_numpy(),
                "rows": np.array([m.shape[0] for m in maps], dtype=np.int32),
                "cols": np.array([m.shape[1] for m in maps], dtype=np.int32),
                "split_raw": [_unwrap(v) for v in sub[split_col]],
                "failure_type_raw": [_unwrap(v) for v in sub["failureType"]],
                "map_bytes": [m.tobytes() for m in maps],
            },
            schema=schema,
        )
        pq.write_table(table, out_dir / f"part-{parts:04d}.parquet", compression="zstd")
        parts += 1
    return {"rows": n, "parts": parts}


def decode_map(map_bytes: bytes, rows: int, cols: int) -> np.ndarray:
    return np.frombuffer(map_bytes, dtype=np.uint8).reshape(rows, cols)
