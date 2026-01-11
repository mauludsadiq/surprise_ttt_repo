from surprise_ttt.gating import BudgetGate, SigmoidGate


def test_budget_gate_clips() -> None:
    g = BudgetGate(eta=1.0, tau=1.0, eps=1e-8)
    assert abs(g(0.0) - 1.0) < 1e-7  # clip to 1
    assert abs(g(1.0) - 1.0) < 1e-7
    assert abs(g(2.0) - 0.5) < 1e-7


def test_sigmoid_gate_range() -> None:
    g = SigmoidGate(eta=0.1, tau=1.0, temp=0.5)
    v = g(1.0)
    assert 0.0 <= v <= 0.1
