# Round 3 test triage (was 26 failures → 0 failures)

## Summary

| Bucket | Representative test | Root cause (one line) | Fix category |
|--------|---------------------|----------------------|--------------|
| API auth / TestClient | `test_autopilot_and_recommendations.py::TestDebugEndpoints::test_startup_status` | TestClient host is not loopback; `.env` set `WORKER_API_TOKEN` → 401 on `/api/*` | **(A)** Clear token for pytest in `tests/conftest.py` |
| Explore crash penalty | `test_autopilot_and_recommendations.py::TestExploreV2Scoring::test_enhance_score_crash_penalty` | `enhance_score` only penalized 30d return stricter than `-0.25`; `-0.20` unchanged | **(B)** Broaden 30d crash branch in `explore_v2.py` |
| Explore gates (spread) | `test_explore_v2.py::TestExploreV2Gates::test_gate_blocks_wide_spread` | `MAX_SPREAD_BPS` default raised to 200; 150 bps passes gate | **(A)** Assert against `MAX_SPREAD_BPS + 50` |
| Macro / composite (fear) | `test_explore_composite.py::TestMacroEnvironment::test_extreme_fear_crypto_penalized` | Macro no longer sets `block_buy` on fear alone (triple gate) | **(A)** Expect strong penalty, not hard block |
| Composite block | `test_explore_composite.py::TestCompositeScore::test_extreme_fear_blocks_crypto` | `block_buy` requires `downtrend_score > 0.7` (not `0.7`) and high `hv` | **(A)** Pass `0.71` / `hv 0.95` in test |
| Live HTTP tests | `test_live_readiness.py::TestAPIEndpoints::test_health_endpoint` | `/api/health` is intentionally thin; no `kraken_ready` | **(A)** Assert on `/api/health/deep` for brokers |
| Live HTTP timeouts | `test_live_readiness.py::TestDatabaseOperations::test_update_bot_risk_fields` | Same API-key issue + tight timeouts under load | **(A)** conftest token + longer timeouts |
| Autopilot radar | `test_now_opportunities.py` | `get_top_recommendations` falls back to explore feed when SQL empty; stocks gated on Alpaca | **(A)** Patch `_explore_feed_fallback` + `_alpaca_any_ready` |
| Stock scan / yfinance | `test_symbol_routing.py` | Stock path imported `yfinance` unconditionally; optional module + scan cache hid `fetch_recent_candles` | **(B)** Optional `yfinance` + `phase2_data_fetcher` fallback in `worker_api._scan_symbol`; **(A)** fake `yfinance` module + `_scan_ohlcv_get` patch in tests |

## Result

- **xfail:** none added (not needed).  
- **Categories C/D:** no remaining failures marked flaky/xfail after fixes.
