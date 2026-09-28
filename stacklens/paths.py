"""프로젝트 경로. 모든 스크립트는 이 모듈을 기준으로 파일을 찾는다."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
BRONZE = DATA / "bronze"
SILVER = DATA / "silver"
GOLD = DATA / "gold"
ARTIFACTS = ROOT / "artifacts"
RESULTS = ARTIFACTS / "results"

LSWMD_PKL = RAW / "LSWMD.pkl"
LSWMD_URL = "https://huggingface.co/datasets/lslattery/wafer-defect-detection/resolve/main/LSWMD.pkl"
LSWMD_SIZE = 2_095_505_977


def ensure_dirs() -> None:
    for p in (RAW, BRONZE, SILVER, GOLD, ARTIFACTS, RESULTS):
        p.mkdir(parents=True, exist_ok=True)
