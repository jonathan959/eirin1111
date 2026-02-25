"""
Phase 4C: Reinforcement Learning Trading Agent
Uses PPO (Proximal Policy Optimization) to learn optimal trading policy.
Requires: pip install gymnasium stable-baselines3
"""
import os
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

ENABLE_RL_AGENT = os.getenv("ENABLE_RL_AGENT", "0").strip().lower() in (
    "1", "true", "yes", "y", "on",
)

RL_MODEL_PATH = os.getenv("RL_MODEL_PATH", "./ml_models/rl_trading_agent")


def _closes(candles: List[List[float]]) -> List[float]:
    return [float(c[4]) for c in candles if len(c) >= 5]


class TradingEnvironment:
    """
    Custom environment for RL trading.
    State: OHLCV + indicators (normalized).
    Actions: 0=hold, 1=buy, 2=sell.
    """

    def __init__(self, data: List[List[float]], initial_balance: float = 10000.0) -> None:
        self.data = data
        self.closes = _closes(data) if data else []
        self.initial_balance = initial_balance
        self.current_step = 0
        self.position = 0  # -1, 0, 1
        self.entry_price = 0.0
        self.balance = initial_balance

    def reset(self) -> List[float]:
        self.current_step = 20
        self.position = 0
        self.balance = self.initial_balance
        return self._get_obs()

    def step(self, action: int) -> tuple:
        if self.current_step >= len(self.closes) - 1:
            return self._get_obs(), 0.0, True, {}
        price = self.closes[self.current_step]
        reward = 0.0
        if action == 1 and self.position == 0:
            self.position = 1
            self.entry_price = price
        elif action == 2 and self.position == 1:
            pnl_pct = (price - self.entry_price) / self.entry_price
            self.balance *= (1 + pnl_pct)
            reward = pnl_pct
            self.position = 0
        self.current_step += 1
        done = self.current_step >= len(self.closes) - 1
        return self._get_obs(), reward, done, {}

    def _get_obs(self) -> List[float]:
        if self.current_step < 20 or self.current_step >= len(self.closes):
            return [0.0] * 20
        c = self.closes
        i = self.current_step
        # Simple features: returns, position
        rets = [(c[i] - c[i - j]) / c[i - j] if c[i - j] else 0 for j in [1, 5, 10]]
        mean_20 = sum(c[i - 20 : i]) / 20 if i >= 20 else c[i]
        std_20 = (sum((x - mean_20) ** 2 for x in c[i - 20 : i]) / 20) ** 0.5 if i >= 20 else 0.01
        z = (c[i] - mean_20) / std_20 if std_20 else 0
        obs = rets + [z, self.position, self.balance / self.initial_balance] + [0.0] * 14
        return obs[:20]


class RLTradingAgent:
    """
    RL agent for trading. Uses PPO when stable-baselines3 available.
    """

    def __init__(self) -> None:
        self.model = None
        self._loaded = False
        if ENABLE_RL_AGENT:
            self._load_model()

    def _load_model(self) -> None:
        try:
            from pathlib import Path
            path = Path(RL_MODEL_PATH)
            if path.exists():
                from stable_baselines3 import PPO
                self.model = PPO.load(str(path))
                self._loaded = True
                logger.info("RL agent model loaded")
        except ImportError:
            logger.debug("stable_baselines3 not installed, RL agent disabled")
        except Exception as e:
            logger.debug("RL model load failed: %s", e)

    def predict_action(self, current_state: List[float]) -> Dict[str, Any]:
        """
        Get optimal action for current state.
        Returns: {action: 'hold'|'buy'|'sell', raw_action: int, confidence: float}
        """
        if not self._loaded or self.model is None:
            return {"action": "hold", "raw_action": 0, "confidence": 0.0, "reason": "RL not loaded"}
        try:
            import numpy as np
            obs = np.array(current_state, dtype=np.float32).reshape(1, -1)
            action, _ = self.model.predict(obs, deterministic=True)
            a = int(action[0])
            action_map = {0: "hold", 1: "buy", 2: "sell"}
            return {"action": action_map.get(a, "hold"), "raw_action": a, "confidence": 0.7}
        except Exception as e:
            logger.debug("RL predict failed: %s", e)
            return {"action": "hold", "raw_action": 0, "confidence": 0.0, "error": str(e)}

    def get_signal_from_candles(self, candles: List[List[float]]) -> Dict[str, Any]:
        """Convert candles to state and get RL signal."""
        env = TradingEnvironment(candles)
        obs = env._get_obs()
        return self.predict_action(obs)
