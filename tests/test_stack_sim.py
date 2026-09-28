"""적층 시뮬레이터가 스스로 맞는지부터 확인한다 (정답을 아는 실험)."""

import numpy as np
import pytest

from stacklens.stack.sim import (
    Policy,
    StackParams,
    closed_form_random_yield,
    evaluate,
    latent_probs,
    sweep_screen,
)

RNG = np.random.default_rng(7)
RISK = RNG.gamma(shape=0.6, scale=1.0, size=60_000)  # 오른쪽 꼬리가 긴 위험 분포


def test_latent_probs_keep_mean_and_bounds():
    for beta in [0, 0.5, 1, 2, 4]:
        p = latent_probs(RISK, 0.01, beta)
        assert p.mean() == pytest.approx(0.01, rel=1e-6)
        assert p.min() >= 0 and p.max() <= 1


def test_beta_zero_is_uniform():
    p = latent_probs(RISK, 0.02, 0)
    assert np.allclose(p, 0.02)


def test_higher_beta_concentrates_latent_in_high_risk():
    top = np.quantile(RISK, 0.95) <= RISK
    shares = [latent_probs(RISK, 0.01, b)[top].sum() / latent_probs(RISK, 0.01, b).sum() for b in [0, 1, 2]]
    assert shares[0] == pytest.approx(0.05, abs=0.005)
    assert shares[0] < shares[1] < shares[2]


def test_random_assembly_matches_closed_form():
    """무작위 조립의 기대 스택 수율 = (E[1-p])^N (표본이 크면)."""
    params = StackParams(n_high=12, escape_rate=0.01, beta=1.0, layer_yield=1.0, base_yield=1.0)
    out = evaluate(RISK, params, Policy(0.0, "random"), n_shuffles=16)
    p = latent_probs(RISK, 0.01, 1.0)
    assert out["stack_yield"] == pytest.approx(closed_form_random_yield(p, 12), rel=0.01)


def test_bernoulli_monte_carlo_agrees_with_expectation():
    """기댓값 계산이 실제로 불량을 뽑아 본 결과와 같은지."""
    params = StackParams(n_high=8, escape_rate=0.02, beta=1.0, layer_yield=1.0, base_yield=1.0)
    risk = RISK[:8_000]
    p = latent_probs(risk, 0.02, 1.0)
    order = np.argsort(risk, kind="stable")
    groups = p[order][: (len(p) // 8) * 8].reshape(-1, 8)
    rng = np.random.default_rng(1)
    sims = [(rng.random(groups.shape) >= groups).all(axis=1).sum() for _ in range(400)]
    expected = evaluate(risk, params, Policy(0.0, "matched"))["good_stacks"]
    assert np.mean(sims) == pytest.approx(expected, rel=0.01)


def test_uniform_latent_screening_never_helps():
    """β=0이면 위험 점수가 잠재불량과 무관하므로 선별은 다이만 버린다."""
    params = StackParams(n_high=16, escape_rate=0.01, beta=0.0)
    res = sweep_screen(RISK, params, "random", objective="good_per_1000_pass")
    assert res["best"]["screen_frac"] == 0.0


def test_matched_assembly_beats_random_when_risk_varies():
    """Π(1-p)는 볼록 → 위험을 한 스택에 몰면 양품 스택 기댓값이 늘어난다."""
    params = StackParams(n_high=12, escape_rate=0.01, beta=1.5)
    r = evaluate(RISK, params, Policy(0.0, "random"))["good_stacks"]
    m = evaluate(RISK, params, Policy(0.0, "matched"))["good_stacks"]
    assert m > r


def test_taller_stack_prefers_more_screening():
    """단수가 높을수록 불량 다이 하나가 버리는 양이 커지므로 최적 선별 비율은 줄지 않는다."""
    qs = []
    for n in [8, 12, 16]:
        params = StackParams(n_high=n, escape_rate=0.01, beta=1.5)
        qs.append(sweep_screen(RISK, params, "random")["best"]["screen_frac"])
    assert qs[0] <= qs[1] <= qs[2]
    assert qs[2] > 0


def test_fast_sweep_matches_evaluate():
    from stacklens.stack.sim import fast_sweep

    params = StackParams(n_high=12, escape_rate=0.01, beta=1.2)
    risk = RISK[:20_000]
    p = latent_probs(risk, 0.01, 1.2)
    qs = np.array([0.0, 0.03, 0.1])
    fast = fast_sweep(risk, p, params, "matched", qs, np.random.default_rng(0))
    for q, row in zip(qs, fast, strict=True):
        slow = evaluate(risk, params, Policy(float(q), "matched"), latent=p)
        assert row["cost"] == pytest.approx(slow["cost_per_good_stack"], rel=1e-9)
        assert row["latent_removed_frac"] == pytest.approx(slow["latent_removed_frac"], rel=1e-9)
