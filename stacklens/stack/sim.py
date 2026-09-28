"""HBM 적층 수율 시뮬레이터 (SIM).

입력은 **웨이퍼 테스트를 통과한 다이**들의 위험 점수 r_i 다(REAL 모델이 실제 웨이퍼맵에서 추정).
테스트를 통과했지만 실제로는 불량인 다이(잠재불량, test escape)가 있다고 보고, 그 확률을

    p_i = e · (r_i / r̄)^β        (평균이 e가 되도록 정규화, 0~1로 자름)

로 둔다. e와 β는 공개되지 않은 값이라 **가정 파라미터**다.
- e: 통과 다이 중 잠재불량 비율
- β: 잠재불량이 "주변에 불량이 많은 곳"에 몰리는 정도. β=0이면 위치와 무관, β=1이면 위험에 비례.
  GDBN(Good Die in Bad Neighborhood)은 β>0 이라는 전제 위에 서 있다.

스택은 코어 다이 N장 + 베이스 다이 1장. 코어 다이가 하나라도 불량이면 스택 전체가 불량이다.
스택 양품 확률 = b · a^N · Π(1 − p_i)   (b: 베이스 다이 수율, a: 층당 조립 수율)

정책은 두 축이다.
- 선별(screening): 위험 상위 q 비율의 통과 다이를 적층 전에 뺀다
- 조립(assembly): 무작위로 묶거나(random), 위험이 비슷한 다이끼리 묶는다(matched)

이 모듈은 기댓값을 닫힌식으로 계산한다. Monte Carlo는 검증(tests/)에서만 쓴다.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class StackParams:
    n_high: int = 12  # 적층 단수 (코어 다이 수)
    escape_rate: float = 0.005  # e: 통과 다이 중 잠재불량 평균 비율 [가정]
    beta: float = 1.0  # β: 잠재불량의 공간 집중도 [가정]
    layer_yield: float = 0.999  # a: 층당 조립 수율 [가정]
    base_yield: float = 0.99  # b: 베이스(로직) 다이 수율 [가정]
    # 비용은 "코어 다이 1장 = 1" 기준 상대값 [가정]
    cost_core_die: float = 1.0
    cost_base_die: float = 3.0
    cost_assembly: float = 4.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Policy:
    screen_frac: float = 0.0  # q: 위험 상위 q를 적층 전에 제외
    assembly: str = "random"  # "random" | "matched"


def latent_probs(risk: np.ndarray, escape_rate: float, beta: float) -> np.ndarray:
    """위험 점수 → 잠재불량 확률. 평균이 escape_rate가 되도록 맞춘다."""
    r = np.asarray(risk, dtype=float)
    if np.any(r < 0):
        raise ValueError("risk는 0 이상이어야 한다")
    if beta == 0 or np.allclose(r, r.mean()):
        return np.full_like(r, escape_rate)
    w = np.power(np.maximum(r, 1e-12) / max(r.mean(), 1e-12), beta)
    p = escape_rate * w / w.mean()
    # 1을 넘는 값은 자르고, 잘린 만큼을 나머지에 재분배해 평균을 유지한다
    for _ in range(20):
        over = p > 1.0
        if not over.any():
            break
        excess = (p[over] - 1.0).sum()
        p[over] = 1.0
        room = ~over
        p[room] += excess * p[room] / p[room].sum()
    return np.clip(p, 0.0, 1.0)


def _group_log_survival(p_sorted: np.ndarray, n: int) -> np.ndarray:
    """연속된 n개씩 묶었을 때 각 묶음의 log Π(1-p). 남는 다이는 버린다."""
    k = len(p_sorted) // n
    if k == 0:
        return np.empty(0)
    lp = np.log1p(-np.minimum(p_sorted[: k * n], 1 - 1e-15))
    return lp.reshape(k, n).sum(axis=1)


def evaluate(
    risk: np.ndarray,
    params: StackParams,
    policy: Policy,
    rng: np.random.Generator | None = None,
    n_shuffles: int = 8,
    latent: np.ndarray | None = None,
) -> dict:
    """통과 다이 집합 하나에 대해 정책의 기대 성과를 계산한다.

    risk    : 선별·매칭에 쓰는 점수 (정책이 보는 값)
    latent  : 각 다이의 잠재불량 확률. 주지 않으면 risk에서 (e, β) 가정으로 만든다.
              REAL-proxy 모드에서는 0/1 정답(실제 불량 다이를 잠재불량으로 삼은 것)을 준다.
    """
    rng = rng or np.random.default_rng(0)
    p = latent_probs(risk, params.escape_rate, params.beta) if latent is None else np.asarray(latent, float)
    n_pass = len(p)
    order = np.argsort(risk, kind="stable")  # 위험 오름차순
    n_keep = int(round(n_pass * (1 - policy.screen_frac)))
    keep = order[:n_keep]
    p_keep = p[keep]
    n = params.n_high

    if policy.assembly == "matched":
        # 위험 오름차순으로 정렬돼 있으므로 이웃끼리 묶으면 위험이 비슷한 다이끼리 묶인다
        logs = [_group_log_survival(p_keep, n)]
    elif policy.assembly == "random":
        logs = [_group_log_survival(rng.permutation(p_keep), n) for _ in range(n_shuffles)]
    else:
        raise ValueError(policy.assembly)

    core_survival = np.mean([np.exp(lg).sum() for lg in logs]) if logs[0].size else 0.0
    n_stacks = n_keep // n
    stack_factor = params.base_yield * params.layer_yield**n
    good = core_survival * stack_factor

    cost = (
        n_pass * params.cost_core_die  # 버린 다이도 이미 만든 비용이다
        + n_stacks * (params.cost_base_die + params.cost_assembly)
    )
    removed_latent = float(p[order[n_keep:]].sum())  # 선별로 걸러낸 잠재불량 기댓값
    return {
        "n_pass": n_pass,
        "n_keep": n_keep,
        "n_stacks": n_stacks,
        "good_stacks": float(good),
        "stack_yield": float(good / n_stacks) if n_stacks else 0.0,
        "good_per_1000_pass": float(good / n_pass * 1000) if n_pass else 0.0,
        "cost_per_good_stack": float(cost / good) if good > 0 else float("inf"),
        "latent_total": float(p.sum()),
        "latent_removed": removed_latent,
        "latent_removed_frac": float(removed_latent / p.sum()) if p.sum() > 0 else 0.0,
        "good_dies_discarded": float((1 - p[order[n_keep:]]).sum()),
    }


def sweep_screen(
    risk: np.ndarray,
    params: StackParams,
    assembly: str,
    fracs: np.ndarray | None = None,
    objective: str = "cost_per_good_stack",
) -> dict:
    """선별 비율 q를 훑어 목적함수 최적값을 찾는다."""
    fracs = np.round(np.arange(0, 0.201, 0.005), 4) if fracs is None else fracs
    rows = [{"screen_frac": float(q), **evaluate(risk, params, Policy(float(q), assembly))} for q in fracs]
    vals = np.array([r[objective] for r in rows])
    best = int(np.argmin(vals) if objective.startswith("cost") else np.argmax(vals))
    return {"rows": rows, "best": rows[best], "baseline": rows[0]}


def fast_sweep(
    score: np.ndarray,
    latent: np.ndarray,
    params: StackParams,
    assembly: str,
    fracs: np.ndarray,
    rng: np.random.Generator,
    n_shuffles: int = 2,
) -> list[dict]:
    """sweep_screen과 같은 값을 정렬 한 번으로 계산한다 (대규모 실험용).

    `evaluate`를 q마다 부르면 매번 전체 정렬을 다시 한다. 여기서는 점수 오름차순으로 한 번 정렬한 뒤
    앞에서부터 n_keep개를 잘라 쓴다. tests/test_stack_sim.py가 두 함수의 결과가 같음을 확인한다.
    """
    order = np.argsort(score, kind="stable")
    p_sorted = np.asarray(latent, float)[order]
    n_pass, n = len(p_sorted), params.n_high
    stack_factor = params.base_yield * params.layer_yield**n
    total_latent = p_sorted.sum()
    rows = []
    for q in fracs:
        n_keep = int(round(n_pass * (1 - q)))
        kept = p_sorted[:n_keep]
        if assembly == "matched":
            surv = np.exp(_group_log_survival(kept, n)).sum()
        else:
            surv = np.mean(
                [np.exp(_group_log_survival(rng.permutation(kept), n)).sum() for _ in range(n_shuffles)]
            )
        n_stacks = n_keep // n
        good = surv * stack_factor
        cost = n_pass * params.cost_core_die + n_stacks * (params.cost_base_die + params.cost_assembly)
        removed = p_sorted[n_keep:].sum()
        rows.append(
            {
                "q": float(q),
                "cost": float(cost / good) if good > 0 else float("inf"),
                "good_per_1000": float(good / n_pass * 1000),
                "stack_yield": float(good / n_stacks) if n_stacks else 0.0,
                "latent_removed_frac": float(removed / total_latent) if total_latent > 0 else 0.0,
            }
        )
    return rows


def closed_form_random_yield(p: np.ndarray, n: int) -> float:
    """복원추출 근사: 무작위로 n장을 뽑을 때 E[Π(1-p)] = (E[1-p])^n."""
    return float(np.mean(1 - p) ** n)
