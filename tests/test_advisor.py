"""advisor 검사기가 실제로 위반을 잡는지 (음성 테스트 포함)."""

from stacklens.advisor.report import (
    Action,
    Fact,
    KgdReport,
    WaferAdvice,
    WaferCase,
    check_fidelity,
    decide,
    render,
)

FACTS = [Fact("P1", "단수", "16단"), Fact("P2", "선별 비율", 0.05, "%"), Fact("W1.3", "불량률", 0.12, "%")]


def _report(reason: str, summary: str = "요약 [P1]") -> KgdReport:
    return KgdReport(
        summary=summary,
        wafers=[
            WaferAdvice(wafer_ref="W1", proposed_action=Action.RELEASE, reason=reason, check_items=["확인"])
        ],
        caveats=["가정에 의존"],
    )


def test_clean_report_passes():
    res = check_fidelity(_report("불량률 [W1.3], 선별 [P2]"), FACTS, ["W1"])
    assert res["ok"] and res["n_citations"] == 3


def test_raw_number_is_caught():
    res = check_fidelity(_report("불량률이 12%로 높다"), FACTS, ["W1"])
    assert not res["ok"] and res["raw_digit_violations"]


def test_fullwidth_digit_is_caught():
    res = check_fidelity(_report("불량률 １２%"), FACTS, ["W1"])
    assert not res["ok"]


def test_unknown_citation_is_caught():
    res = check_fidelity(_report("불량률 [W9.3]"), FACTS, ["W1"])
    assert res["bad_citations"] == ["W9.3"]


def test_missing_wafer_is_caught():
    res = check_fidelity(_report("[P2]"), FACTS, ["W1", "W2"])
    assert res["missing_wafers"] == ["W2"] and not res["ok"]


def test_render_substitutes_values():
    assert render("[P1] 제품, 선별 [P2]", FACTS) == "16단 제품, 선별 5.0%"


def test_rule_engine():
    base = dict(wafer_id=1, pattern_prob=0.9, n_pass=500)
    assert (
        decide(
            WaferCase("W", pattern_pred="Near-full", fail_rate=0.3, n_high_risk=0, high_risk_share=0, **base)
        )
        == Action.SCRAP_CANDIDATE
    )
    assert (
        decide(WaferCase("W", pattern_pred="none", fail_rate=0.6, n_high_risk=0, high_risk_share=0, **base))
        == Action.SCRAP_CANDIDATE
    )
    assert (
        decide(
            WaferCase("W", pattern_pred="Loc", fail_rate=0.1, n_high_risk=200, high_risk_share=0.4, **base)
        )
        == Action.HOLD_REVIEW
    )
    assert (
        decide(WaferCase("W", pattern_pred="Loc", fail_rate=0.1, n_high_risk=3, high_risk_share=0.01, **base))
        == Action.SCREEN_HIGH_RISK
    )
    assert (
        decide(WaferCase("W", pattern_pred="none", fail_rate=0.02, n_high_risk=0, high_risk_share=0, **base))
        == Action.RELEASE
    )


def test_wafer_names_are_not_numbers():
    """검사기 v1 오탐 회귀 방지: 'W10', '[W9]'처럼 웨이퍼를 부르는 것은 위반이 아니다."""
    rep = _report("W1은 [W1.3] 조건으로 검토, [W1] 참조")
    res = check_fidelity(rep, FACTS, ["W1"])
    assert res["ok"], res


def test_digits_next_to_wafer_name_still_caught():
    rep = _report("W1 불량률 4.3%")
    assert not check_fidelity(rep, FACTS, ["W1"])["ok"]
