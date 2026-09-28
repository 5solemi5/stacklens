"""실행 계보(lineage) 기록.

"이 결과가 어떤 입력에서, 어떤 코드로, 언제 나왔는가"를 파일 하나로 남긴다.
입력 파일은 SHA256, 출력은 행 수와 파일 목록, 코드는 git 커밋.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path


def sha256(path: Path, block: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(block):
            h.update(chunk)
    return h.hexdigest()


def git_commit(root: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        )
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True)
        return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def write_manifest(
    path: Path, stage: str, inputs: dict, outputs: dict, root: Path, extra: dict | None = None
):
    record = {
        "stage": stage,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(root),
        "python": platform.python_version(),
        "inputs": inputs,
        "outputs": outputs,
    }
    if extra:
        record.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2))
    return record
