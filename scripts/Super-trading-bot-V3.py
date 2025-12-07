# File: Super-trading-bot-V3.py
"""
Super-trading-bot-V3
Strategy: Channel breakout with ATR stop-loss / take-profit.
"""

from typing import List, Literal, Union

Price = Union[float, int]
Signal = Literal["BUY", "SELL", "FLAT"]


def true_range(high: List[Price], low: List[Price], close: List[Price]) -> List[float]:
    tr: List[float] = []
    for i in range(len(close)):
        if i == 0:
            tr.append(high[i] - low[i])
        else:
            tr.append(
                max(
                    high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i] - close[i - 1]),
                )
            )
    return tr


def atr(high: List[Price], low: List[Price], close: List[Price], period: int = 14) -> List[float]:
    tr = true_range(high, low, close)
    atr_values: List[float] = []
    running = 0.0

    for i, v in enumerate(tr):
        running += v
        if i < period:
            atr_values.append(float("nan"))
            continue
        if i == period:
            atr_values.append(running / period)
        else:
            prev = atr_values[-1]
            atr_values.append((prev * (period - 1) + v) / period)

    return atr_values


def highest(values: List[Price], period: int) -> List[float]:
    out: List[float] = []
    for i in range(len(values)):
        if i + 1 < period:
            out.append(float("nan"))
        else:
            window = values[i + 1 - period : i + 1]
            out.append(max(window))
    return out


def lowest(values: List[Price], period: int) -> List[float]:
    out: List[float] = []
    for i in range(len(values)):
        if i + 1 < period:
            out.append(float("nan"))
        else:
            window = values[i + 1 - period : i + 1]
            out.append(min(window))
    return out


def generate_signals(
    high: List[Price],
    low: List[Price],
    close: List[Price],
    channel_period: int = 20,
) -> List[Signal]:
    hi = highest(high, channel_period)
    lo = lowest(low, channel_period)

    signals: List[Signal] = []

    for h, l, c in zip(hi, lo, close):
        if any(map(lambda x: x != x, (h, l))):
            signals.append("FLAT")
            continue

        if c > h:
            signals.append("BUY")
        elif c < l:
            signals.append("SELL")
        else:
            signals.append("FLAT")

    return signals


def backtest(
    high: List[Price],
    low: List[Price],
    close: List[Price],
    channel_period: int = 20,
    atr_period: int = 14,
    atr_tp: float = 3.0,
    atr_sl: float = 1.5,
) -> float:
    signals = generate_signals(high, low, close, channel_period)
    atr_values = atr(high, low, close, atr_period)

    position = 0
    entry_price = None
    pnl = 0.0

    for i in range(len(close)):
        c = close[i]
        sig = signals[i]
        a = atr_values[i]

        if a != a:  # NaN atr
            continue

        if position == 0:
            if sig == "BUY":
                position = 1
                entry_price = c
            elif sig == "SELL":
                position = -1
                entry_price = c
        else:
            distance = c - entry_price
            if position == 1:
                if distance >= atr_tp * a or distance <= -atr_sl * a:
                    pnl += distance
                    position = 0
                    entry_price = None
            elif position == -1:
                if -distance >= atr_tp * a or -distance <= -atr_sl * a:
                    pnl += -distance
                    position = 0
                    entry_price = None

    if position != 0 and entry_price is not None:
        last = close[-1]
        pnl += (last - entry_price) * position

    return pnl


if _name_ == "_main_":
    # Dummy OHLC data
    close = [100 + (i % 10) for i in range(200)]
    high = [c + 1 for c in close]
    low = [c - 1 for c in close]
    print("V3 PnL:", backtest(high, low, close))
