"""M3 · HBM 적층 실험.

모드 A — REAL-proxy (순환 없는 정책 비교)
    실제 test 웨이퍼에서 **진짜 불량 다이 일부를 "테스트를 통과한 척하는 잠재불량"으로** 섞는다.
    잠재불량의 위치가 실제 불량의 공간 분포를 그대로 따르므로(= GDBN의 전제),
    정답이 어떤 모델의 점수에서 나온 것이 아니다. 그래서 GDBN 규칙 · NNR · LGBM을 공정하게 비교할 수 있다.
    각 다이의 점수는 교차 마스킹으로 자기 결과를 보지 않고 계산된 값이다.
    한계: 이웃 특징은 섞인 잠재불량을 "불량"으로 봤다(실제로는 양품처럼 보였을 것). 잠재불량 비율이
    통과 다이의 0.2~2%라 영향은 작지만 결과를 약간 낙관적으로 만든다 → docs/limitations.md

모드 B — β 민감도 (가정이 틀리면 결론이 어떻게 바뀌나)
    잠재불량이 위험 점수에 얼마나 몰리는지(β)와 비율(e)을 격자로 바꿔 최적 선별 비율과 이득을 본다.
    β=0(위치와 무관)이면 선별은 다이만 버린다. 이 경계를 숨기지 않는다.
"""

from __future__ import annotations

import json
import time

import numpy as np

from stacklens import paths
from stacklens.stack.sim import StackParams, fast_sweep, latent_probs

HEIGHTS = [8, 12, 16]
ESCAPE_RATES = [0.002, 0.005, 0.01, 0.02]
Q_GRID = np.round(np.r_[np.arange(0, 0.10, 0.0025), np.arange(0.10, 0.301, 0.01)], 4)
SEEDS = [0, 1, 2]
N_WAFERS_A = 1200
POLICY_SCORES = {
    "random": "무작위 선별",
    "B1": "GDBN 규칙(8이웃)",
    "B2": "NNR(5×5)",
    "B4": "LGBM 위험모델",
    "oracle": "오라클(정답)",
}


def sweep(score, latent, params, assembly, rng) -> dict:
    rows = fast_sweep(score, latent, params, assembly, Q_GRID, rng)
    best = min(rows, key=lambda x: x["cost"])
    return {"rows": rows, "best": best, "q0": rows[0]}


def mode_a() -> dict:
    d = np.load(paths.GOLD / "test_die_scores.npz")
    y, wid = d["y"], d["wafer_id"]
    rng0 = np.random.default_rng(0)
    keep_w = rng0.choice(np.unique(wid), N_WAFERS_A, replace=False)
    sel = np.isin(wid, keep_w)
    y = y[sel]
    scores = {k: d[k][sel] for k in ["B1", "B2", "B4"]}
    pass_idx = np.nonzero(y == 0)[0]
    fail_idx = np.nonzero(y == 1)[0]
    out = {"n_wafers": N_WAFERS_A, "n_pass": int(len(pass_idx)), "n_fail": int(len(fail_idx)), "cells": []}
    for e in ESCAPE_RATES:
        n_esc = int(round(e / (1 - e) * len(pass_idx)))
        for seed in SEEDS:
            rng = np.random.default_rng(100 + seed)
            esc = rng.choice(fail_idx, n_esc, replace=False)
            pool = np.concatenate([pass_idx, esc])
            latent = np.r_[np.zeros(len(pass_idx)), np.ones(n_esc)]
            jitter = rng.random(len(pool)) * 1e-6  # 동점(GDBN 0개 이웃 등)은 무작위로 푼다
            pol_scores = {
                "random": rng.random(len(pool)),
                **{k: scores[k][pool] + jitter for k in ["B1", "B2", "B4"]},
                "oracle": latent + jitter,
            }
            for n in HEIGHTS:
                params = StackParams(n_high=n)
                for assembly in ["random", "matched"]:
                    for name, sc in pol_scores.items():
                        if assembly == "matched" and name == "random":
                            continue
                        res = sweep(sc, latent, params, assembly, rng)
                        out["cells"].append(
                            {
                                "escape_rate": e,
                                "seed": seed,
                                "n_high": n,
                                "assembly": assembly,
                                "policy": name,
                                "best_q": res["best"]["q"],
                                "best_cost": res["best"]["cost"],
                                "q0_cost": res["q0"]["cost"],
                                "best_stack_yield": res["best"]["stack_yield"],
                                "q0_stack_yield": res["q0"]["stack_yield"],
                                "latent_removed_at_best": res["best"]["latent_removed_frac"],
                                "curve": [(r["q"], r["cost"]) for r in res["rows"]] if seed == 0 else None,
                            }
                        )
            print(f"  e={e:.3f} seed={seed} 완료")
    return out


def mode_b() -> dict:
    d = np.load(paths.GOLD / "passing_die_risk.npz")
    risk = d["risk"]
    rng = np.random.default_rng(0)
    risk = rng.choice(risk, 600_000, replace=False)
    out = {"n_pass_sample": int(len(risk)), "cells": []}
    for beta in [0.0, 0.5, 1.0, 1.5, 2.0]:
        for e in ESCAPE_RATES:
            for n in HEIGHTS:
                params = StackParams(n_high=n, escape_rate=e, beta=beta)
                for assembly in ["random", "matched"]:
                    res = sweep(risk, latent_probs(risk, e, beta), params, assembly, rng)
                    out["cells"].append(
                        {
                            "beta": beta,
                            "escape_rate": e,
                            "n_high": n,
                            "assembly": assembly,
                            "best_q": res["best"]["q"],
                            "best_cost": res["best"]["cost"],
                            "q0_cost": res["q0"]["cost"],
                        }
                    )
        print(f"  β={beta} 완료")
    return out


COST_SCENARIOS = {"저가 패키지": (1.0, 1.0), "기준": (3.0, 4.0), "고가 패키지": (6.0, 10.0)}


def mode_c() -> dict:
    """비용 가정 민감도: 베이스 다이·조립 비용이 달라지면 최적 선별과 이득이 어떻게 바뀌나 (16단, e=1%)."""
    d = np.load(paths.GOLD / "test_die_scores.npz")
    y, wid = d["y"], d["wafer_id"]
    keep_w = np.random.default_rng(0).choice(np.unique(wid), N_WAFERS_A, replace=False)
    sel = np.isin(wid, keep_w)
    y, b4 = y[sel], d["B4"][sel]
    pass_idx, fail_idx = np.nonzero(y == 0)[0], np.nonzero(y == 1)[0]
    out = []
    for seed in SEEDS:
        rng = np.random.default_rng(100 + seed)
        esc = rng.choice(fail_idx, int(round(0.01 / 0.99 * len(pass_idx))), replace=False)
        pool = np.concatenate([pass_idx, esc])
        latent = np.r_[np.zeros(len(pass_idx)), np.ones(len(esc))]
        score = b4[pool] + rng.random(len(pool)) * 1e-6
        for name, (cb, ca) in COST_SCENARIOS.items():
            params = StackParams(n_high=16, cost_base_die=cb, cost_assembly=ca)
            base = sweep(rng.random(len(pool)), latent, params, "random", rng)["q0"]["cost"]
            for assembly in ["random", "matched"]:
                res = sweep(score, latent, params, assembly, rng)
                out.append(
                    {
                        "scenario": name,
                        "cost_base": cb,
                        "cost_asm": ca,
                        "assembly": assembly,
                        "seed": seed,
                        "best_q": res["best"]["q"],
                        "gain_vs_base": 1 - res["best"]["cost"] / base,
                    }
                )
    return {"n_high": 16, "escape_rate": 0.01, "cells": out}


def main() -> None:
    t0 = time.time()
    print("모드 A: REAL-proxy 정책 비교…")
    a = mode_a()
    (paths.RESULTS / "stack_realproxy.json").write_text(json.dumps(a, ensure_ascii=False))
    print(f"  ({time.time() - t0:.0f}s)")
    print("모드 B: β 민감도…")
    b = mode_b()
    (paths.RESULTS / "stack_sensitivity.json").write_text(json.dumps(b, ensure_ascii=False))
    print("모드 C: 비용 가정 민감도…")
    c = mode_c()
    (paths.RESULTS / "stack_cost_sensitivity.json").write_text(json.dumps(c, ensure_ascii=False))
    print(f"완료 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
