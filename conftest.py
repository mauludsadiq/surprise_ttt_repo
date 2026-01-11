import time

_DESCRIPTIONS = {
    "tests/test_ema.py::test_ema_initializes_to_first_value": "EMA initializes to first observed surprise value (no biased warm start).",
    "tests/test_ema.py::test_ema_updates_correctly": "EMA recursion matches s~_{t+1} = α s~_t + (1-α) s_t exactly.",
    "tests/test_gate.py::test_budget_gate_clips": "Budget gate clips step: gamma = η * clip(τ/(s+ε), 0, 1).",
    "tests/test_gate.py::test_sigmoid_gate_range": "Sigmoid gate stays bounded: 0 ≤ gamma ≤ η for all inputs.",
    "tests/test_quadratic_stability.py::test_non_osc_contracts_in_quadratic": "Quadratic stability: with γ ≤ 1/(λ_max(H)+λ), updates contract without oscillation.",
    "tests/test_updater_smoke.py::test_updater_runs_and_changes_params": "End-to-end updater: computes loss/grad/surprise/gate/retention and changes parameters.",
}

_START = {}

def _desc(nodeid: str) -> str:
    return _DESCRIPTIONS.get(nodeid, nodeid)

def pytest_runtest_setup(item):
    _START[item.nodeid] = time.perf_counter()
    print(f"TEST  {_desc(item.nodeid)}", flush=True)

def pytest_runtest_logreport(report):
    if report.when != "call":
        return
    dt = None
    if report.nodeid in _START:
        dt = time.perf_counter() - _START[report.nodeid]
    suffix = "" if dt is None else f"  ({dt:.3f}s)"
    if report.passed:
        print(f"PASS  {_desc(report.nodeid)}{suffix}", flush=True)
    elif report.failed:
        print(f"FAIL  {_desc(report.nodeid)}{suffix}", flush=True)
    elif report.skipped:
        print(f"SKIP  {_desc(report.nodeid)}{suffix}", flush=True)
