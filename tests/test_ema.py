from surprise_ttt.ema import EMA


def test_ema_initializes_to_first_value() -> None:
    e = EMA(alpha=0.9)
    v1 = e.update(10.0)
    assert v1 == 10.0
    assert e.initialized


def test_ema_updates_correctly() -> None:
    e = EMA(alpha=0.5)
    e.update(10.0)  # init
    v2 = e.update(14.0)
    # 0.5*10 + 0.5*14 = 12
    assert abs(v2 - 12.0) < 1e-9
