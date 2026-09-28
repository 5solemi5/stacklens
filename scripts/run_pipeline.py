"""원본 → bronze → silver, 품질 계약 리포트와 manifest를 남긴다.

python scripts/run_pipeline.py            # 전체
python scripts/run_pipeline.py --skip-bronze
"""

from __future__ import annotations

import argparse
import json
import time

import duckdb

from stacklens import paths
from stacklens.pipeline import ingest, layers
from stacklens.pipeline.manifest import sha256, write_manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-bronze", action="store_true")
    args = ap.parse_args()
    paths.ensure_dirs()

    if not paths.LSWMD_PKL.exists():
        raise SystemExit(
            f"원본이 없습니다: {paths.LSWMD_PKL}\n  curl -L -o {paths.LSWMD_PKL} {paths.LSWMD_URL}"
        )
    size = paths.LSWMD_PKL.stat().st_size
    if size != paths.LSWMD_SIZE:
        raise SystemExit(f"원본 크기 불일치: {size} != {paths.LSWMD_SIZE} (다운로드가 덜 끝났을 수 있음)")

    t0 = time.time()
    bronze_dir = paths.BRONZE / "wafers"
    if not args.skip_bronze:
        print("[1/3] 원본 pickle 읽는 중 (1~2분)…")
        df = ingest.load_lswmd(paths.LSWMD_PKL)
        print(f"      {len(df):,}행 · 컬럼 {list(df.columns)}")
        info = ingest.to_bronze(df, bronze_dir)
        del df
        write_manifest(
            paths.BRONZE / "manifest.json",
            "bronze",
            {"LSWMD.pkl": {"sha256": sha256(paths.LSWMD_PKL), "bytes": size, "source": paths.LSWMD_URL}},
            {"wafers": info},
            paths.ROOT,
        )
        print(f"      bronze {info['rows']:,}행 · {info['parts']}개 파티션 ({time.time() - t0:.0f}s)")

    print("[2/3] silver 변환 + 품질 계약 검사…")
    report = layers.bronze_to_silver(bronze_dir, paths.SILVER / "wafers")

    print("[3/3] 요약 집계(DuckDB)…")
    con = duckdb.connect()
    glob = str(paths.SILVER / "wafers" / "part-*.parquet")
    summary = con.execute(
        f"""
        select role, count(*) wafers, count(distinct lot_name) lots,
               sum(labeled::int) labeled, sum(n_dies) dies, avg(fail_rate) mean_fail_rate
        from read_parquet('{glob}') group by role order by role
        """
    ).df()
    patterns = con.execute(
        f"select pattern, count(*) n from read_parquet('{glob}') where labeled group by 1 order by 2 desc"
    ).df()
    report["summary_by_role"] = summary.to_dict(orient="records")
    report["pattern_counts"] = patterns.to_dict(orient="records")

    write_manifest(
        paths.SILVER / "manifest.json",
        "silver",
        {"bronze": str(bronze_dir.relative_to(paths.ROOT))},
        {"wafers": {"rows": report["rows_out"]}},
        paths.ROOT,
    )
    out = paths.RESULTS / "pipeline_quality.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=float))

    print(
        f"\n입력 {report['rows_in']:,} → silver {report['rows_out']:,} (제외 사유 {report['dropped_by_reason']})"
    )
    for c in report["contracts"]:
        mark = "PASS" if c["passed"] else ("FAIL" if c["severity"] == "block" else "WARN")
        print(f"  [{mark}] {c['name']:28s} {c['n_failed']:>8,} / {c['n_checked']:>8,}  {c['description']}")
    print(summary.to_string(index=False))
    print(patterns.to_string(index=False))
    print(f"완료 {time.time() - t0:.0f}s → {out.relative_to(paths.ROOT)}")


if __name__ == "__main__":
    main()
