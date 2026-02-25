import sys
import os
sys.path.append(os.getcwd())
from strategies import select_strategy, RegimeResult

regime = RegimeResult(regime="TREND_UP", confidence=0.8, why=[], snapshot={})
target, switched, why = select_strategy(regime, current="smart", last_switch_ts=0, now_ts=100, min_hold_sec=60)

print(f"Regime: {regime.regime}")
print(f"Target: {target}")
print(f"Switched: {switched}")
print(f"Why: {why}")
