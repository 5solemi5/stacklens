"""모든 실험 결과 → artifacts/results/summary.json (README·앱·포트폴리오가 읽는 단일 원천)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from stacklens import paths

R = paths.RESULTS


def load(name):
    return json.loads((R / name).read_text())


def stack_table() -> dict:
    a = load("stack_realproxy.json")
    df = pd.DataFrame(a["cells"])
    base = df[(df.policy == "random") & (df.assembly == "random")][
        ["escape_rate", "seed", "n_high", "q0_cost", "q0_stack_yield"]
    ]
    base = base.rename(columns={"q0_cost": "base_cost", "q0_stack_yield": "base_yield"})
    df = df.merge(base, on=["escape_rate", "seed", "n_high"])
    df["gain"] = 1 - df.best_cost / df.base_cost
    g = (
        df.groupby(["escape_rate", "n_high", "assembly", "policy"])
        .agg(
            best_q=("best_q", "mean"),
            gain=("gain", "mean"),
            gain_sd=("gain", "std"),
            yield_best=("best_stack_yield", "mean"),
            yield_base=("base_yield", "mean"),
            latent_removed=("latent_removed_at_best", "mean"),
        )
        .reset_index()
    )
    curves = [
        {k: c[k] for k in ["escape_rate", "n_high", "assembly", "policy", "curve"]}
        for c in a["cells"]
        if c["curve"] is not None
    ]
    return {
        "n_wafers": a["n_wafers"],
        "n_pass": a["n_pass"],
        "n_fail": a["n_fail"],
        "rows": g.round(5).to_dict(orient="records"),
        "curves": curves,
    }


def effective_beta(stack_rows: list[dict]) -> dict:
    """REAL-proxy 이득(B4, 무작위 조립)과 같은 이득을 내는 β를 모드 B 격자에서 보간한다."""
    b = pd.DataFrame(load("stack_sensitivity.json")["cells"])
    b["gain"] = 1 - b.best_cost / b.q0_cost
    out = []
    a = pd.DataFrame(stack_rows)
    for e in [0.005, 0.01, 0.02]:
        for n in [8, 12, 16]:
            target = a[
                (a.escape_rate == e) & (a.n_high == n) & (a.assembly == "random") & (a.policy == "B4")
            ].gain
            if target.empty:
                continue
            curve = b[(b.escape_rate == e) & (b.n_high == n) & (b.assembly == "random")].sort_values("beta")
            t = float(target.iloc[0])
            beta = float(np.interp(t, curve.gain, curve.beta)) if curve.gain.max() >= t else float("nan")
            out.append({"escape_rate": e, "n_high": n, "realproxy_gain": t, "effective_beta": beta})
    return {"grid": b.round(5).to_dict(orient="records"), "effective_beta": out}


def main() -> None:
    pipe = load("pipeline_quality.json")
    pat = load("pattern.json")
    risk = load("risk.json")
    stack = stack_table()
    sens = effective_beta(stack["rows"])
    cost = pd.DataFrame(load("stack_cost_sensitivity.json")["cells"])
    cost_g = (
        cost.groupby(["scenario", "assembly"])
        .agg(best_q=("best_q", "mean"), gain=("gain_vs_base", "mean"), gain_sd=("gain_vs_base", "std"))
        .reset_index()
    )

    def pick(e, n, asm, pol):
        return next(
            r
            for r in stack["rows"]
            if r["escape_rate"] == e and r["n_high"] == n and r["assembly"] == asm and r["policy"] == pol
        )

    headline = {
        "wafers": pipe["rows_in"],
        "dies": int(sum(r["dies"] for r in pipe["summary_by_role"])),
        "pattern_macro_f1": pat["lot_split_balanced"]["macro_f1_mean"],
        "pattern_macro_f1_sd": pat["lot_split_balanced"]["macro_f1_std"],
        "pattern_all_none_f1": pat["baseline_all_none_macro_f1"],
        "risk_within_wafer_auc": {k: v["within_wafer_auc"] for k, v in risk["models"].items()},
        "risk_capture_top5": {k: v["capture_top5pct"] for k, v in risk["models"].items()},
        "leak_check_max_delta": risk["leak_check"]["max_abs_pred_delta_when_masked_values_flipped"],
        "stack_16_e1_B4_matched": pick(0.01, 16, "matched", "B4"),
        "stack_16_e1_B1_matched": pick(0.01, 16, "matched", "B1"),
        "stack_16_e1_oracle": pick(0.01, 16, "matched", "oracle"),
        "gain_by_height_B4_matched_e1": {n: pick(0.01, n, "matched", "B4")["gain"] for n in [8, 12, 16]},
    }
    adv = json.loads((paths.ARTIFACTS / "advisor" / "results.json").read_text())
    advisor_table = []
    for ver, models in adv["versions"].items():
        for mdl, r in models.items():
            advisor_table.append(
                {
                    "version": ver,
                    "model": mdl,
                    "fidelity_ok_rate": r["fidelity_ok_rate"],
                    "raw_digit_violations": r["raw_digit_violations"],
                    "bad_citations": r["bad_citations"],
                    "agreement_mean": r["agreement_mean"],
                    "repeat_consistency": r["repeat_consistency"],
                    "injection_flipped": sum(x["flipped_to_release"] for x in r["injection"]),
                    "injection_trials": len(r["injection"]),
                    "latency_mean_s": r["latency_mean_s"],
                    "tokens_out_mean": r["tokens_out_mean"],
                }
            )
    summary = {
        "headline": headline,
        "pipeline": pipe,
        "pattern": {k: pat[k] for k in pat if k not in ("features",)},
        "risk": risk,
        "stack": stack,
        "sensitivity": sens,
        "cost_sensitivity": cost_g.round(5).to_dict(orient="records"),
        "advisor": {"prompt_hash": adv["prompt_hash"], "table": advisor_table},
    }
    (R / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=float))
    h = headline
    print(
        json.dumps(
            {k: v for k, v in h.items() if not isinstance(v, dict) or "gain" in str(v)},
            ensure_ascii=False,
            indent=1,
            default=float,
        )[:3000]
    )
    print(pd.DataFrame(sens["effective_beta"]).round(3).to_string(index=False))
    print(cost_g.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
