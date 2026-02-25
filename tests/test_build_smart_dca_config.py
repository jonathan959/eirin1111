"""
Test build_smart_dca_config (unit test).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_build_smart_dca_config_from_dict():
    from strategies import build_smart_dca_config, SmartDcaConfig
    cfg = build_smart_dca_config({"base_quote": 25, "safety_quote": 15, "tp": 0.03})
    assert isinstance(cfg, SmartDcaConfig)
    assert cfg.base_quote == 25
    assert cfg.safety_quote == 15
    assert cfg.tp == 0.03


def test_build_smart_dca_config_overrides():
    from strategies import build_smart_dca_config
    cfg = build_smart_dca_config({"base_quote": 20}, overrides={"base_quote_mult": 0.5})
    assert cfg.base_quote_mult == 0.5


if __name__ == "__main__":
    test_build_smart_dca_config_from_dict()
    test_build_smart_dca_config_overrides()
    print("test_build_smart_dca_config passed")
