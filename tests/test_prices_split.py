"""Split /api/prices fetch: timeouts, partial merge, TTL cache."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import patch

import pytest

import services.prices_fetch as pf


def _btc_batch():
    return {"BTC/USD": {"last": 50000.0, "percentage": 1.0, "quoteVolume": 100.0}}


def _slow_stocks(_batch):
    time.sleep(6.0)
    return {"snapshots": {}}


def _fast_stocks(batch):
    sym = batch[0] if batch else "AAPL"
    return {
        "snapshots": {
            sym: {"latestTrade": {"p": 222.0}, "dailyBar": {"c": 222.0, "v": 1}, "prevDailyBar": {"c": 200.0}},
        }
    }


def test_partial_when_stocks_timeout_fast_crypto():
    pf.clear_symbol_cache_for_tests()
    pf.reset_upstream_counters()

    def split(req, mt):
        return ["BTC/USD"], ["AAPL"]

    async def _run():
        with patch.object(pf, "normalize_fn", lambda s: s), patch.object(pf, "resolve_fn", lambda s: s), patch.object(
            pf, "split_buckets_fn", split
        ), patch.object(pf, "kraken_batch_fn", _btc_batch), patch.object(pf, "stocks_snapshots_fn", _slow_stocks):
            t0 = time.time()
            out = await pf.fetch_prices_async("BTC/USD,AAPL", "all", timeout_sec=0.4)
            elapsed = time.time() - t0
        assert elapsed < 3.0
        assert out.get("partial") is True
        assert out["prices"].get("BTC/USD") == pytest.approx(50000.0)
        assert "errors" in out

    asyncio.run(_run())


def test_merged_when_both_fast():
    pf.clear_symbol_cache_for_tests()
    pf.reset_upstream_counters()

    def split(req, mt):
        return ["BTC/USD"], ["AAPL"]

    async def _run():
        with patch.object(pf, "normalize_fn", lambda s: s), patch.object(pf, "resolve_fn", lambda s: s), patch.object(
            pf, "split_buckets_fn", split
        ), patch.object(pf, "kraken_batch_fn", _btc_batch), patch.object(pf, "stocks_snapshots_fn", _fast_stocks):
            out = await pf.fetch_prices_async("BTC/USD,AAPL", "all", timeout_sec=2.0)
        assert out.get("partial") is False
        assert out["prices"].get("BTC/USD") == pytest.approx(50000.0)
        assert out["prices"].get("AAPL") == pytest.approx(222.0)

    asyncio.run(_run())


def test_cache_skips_second_kraken_batch():
    pf.clear_symbol_cache_for_tests()
    pf.reset_upstream_counters()
    calls = {"n": 0}

    def counted_batch():
        calls["n"] += 1
        return _btc_batch()

    def split_one(req, mt):
        return ["BTC/USD"], []

    async def _run():
        with patch.object(pf, "normalize_fn", lambda s: s), patch.object(pf, "resolve_fn", lambda s: s), patch.object(
            pf, "split_buckets_fn", split_one
        ), patch.object(pf, "kraken_batch_fn", counted_batch):
            await pf.fetch_prices_async("BTC/USD", "crypto", timeout_sec=2.0)
            await pf.fetch_prices_async("BTC/USD", "crypto", timeout_sec=2.0)

    asyncio.run(_run())
    assert calls["n"] == 1


def test_coingecko_cro_mapping():
    from services.icon_map import SYMBOL_TO_COINGECKO_ID

    assert SYMBOL_TO_COINGECKO_ID["CRO"] == "crypto-com-chain"
    assert "cronos" not in SYMBOL_TO_COINGECKO_ID["CRO"]
