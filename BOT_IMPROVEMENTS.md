# Bot improvement recommendations

Summary of **recommendations** and **code changes** applied to make the bot safer and more tunable. You deploy; no auto-deploy is run from here.

---

## Implemented in code

1. **Per-bot trailing stop params**  
   The run loop now uses each bot’s `trailing_activation_pct` and `trailing_distance_pct` (from the DB) when checking the trailing stop. Previously only env vars were used. You can tune activation/distance per bot in the UI or API.

2. **Bot config validation**  
   On bot create/update (API), critical numeric fields are clamped to safe ranges so typos don’t create unusable bots:
   - `first_dev` 0.001–0.5, `step_mult` 1–10, `tp` 0.001–0.5  
   - `daily_loss_limit_pct` 0.01–0.25, `max_drawdown_pct` 0–0.99  
   - `trailing_activation_pct`, `trailing_distance_pct`, `spread_guard_pct`, gap %, `max_safety` 1–20, `poll_seconds` 5–300  

3. **max_drawdown_pct = 0 means disabled**  
   When a bot’s `max_drawdown_pct` is 0, drawdown kill-switch is not applied (circuit breaker and intelligence layer). Previously 0 could be treated as 20%. Set a value &gt; 0 (e.g. 0.10 or 0.20) to enable the limit.

4. **Per-bot drawdown in intelligence**  
   The intelligence layer’s market-safety gate now uses the bot’s `max_drawdown_pct` when it is set and &gt; 0; otherwise it falls back to the layer default (`INTEL_MAX_DRAWDOWN_PCT`).

---

## Further recommendations (optional)

- **Trailing stop**: Prefer DB fields per bot; use env vars only as fallback. Already done.
- **Daily loss**: Ensure `daily_loss_limit_pct` is set per bot (e.g. 0.05–0.08). Validation now keeps it in 0.01–0.25.
- **Cooldown after stop loss**: Phase1 `stop_loss_cooldown_sec` is already used; consider raising it (e.g. 3600–7200) on volatile symbols to avoid revenge entries.
- **Time filters**: For stocks, keep `time_filter_enabled` and skip first/last 30 min so the bot avoids open/close volatility.
- **BTC correlation**: For alts, keep `btc_correlation_guard` on so the bot pauses when BTC dumps.
- **Symbol selection**: Use the momentum-filtered universe (`RECO_MOMENTUM_FILTER=1`) so recommendations and scans focus on stronger names.
- **Position sizing**: Use Kelly or dynamic sizing (already available) and keep `kelly_fraction` conservative (e.g. 0.25) until you have enough history.
- **Exploration**: Keep `EXPLORE_V2=1` so volume/spread/volatility gates and score penalties apply to recommendations.

---

## Env / config quick reference

| Env / setting            | Purpose |
|-------------------------|--------|
| `ALLOW_LIVE_TRADING`    | Must be 1 for real orders when bot is live (`dry_run=0`). |
| `RECO_MOMENTUM_FILTER`  | 1 = filter stock universe by momentum before scanning. |
| `EXPLORE_V2`            | 1 = apply volume/spread/volatility gates and score penalties. |
| `INTEL_MAX_DRAWDOWN_PCT`| Default drawdown limit when bot’s `max_drawdown_pct` is 0. |
| Bot `max_drawdown_pct`  | 0 = disabled; &gt; 0 = block new trades when portfolio drawdown ≥ this. |
