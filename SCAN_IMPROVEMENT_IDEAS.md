# Ideas: Better Stock/Crypto Scanning & Selection

Ways to improve how the bot finds and ranks the "best possible" stocks and cryptos. You can pick which to implement; below is the full list, then a suggested order.

---

## 1. **Scan profile: Aggressive vs Conservative**

| Idea | What it does |
|------|----------------|
| **Conservative** | Stricter gates: higher min score to be "eligible", higher min volume, tighter spread, avoid high volatility. Fewer but higher-quality names. |
| **Aggressive** | Looser gates: lower score threshold, more volatile names allowed, larger universe. More ideas, higher risk. |
| **Implementation** | Env or setting: `RECO_PROFILE=conservative \| balanced \| aggressive`. This drives: (a) min score threshold (e.g. 50 / 40 / 35), (b) RECO_BUY_THRESHOLD_STOCKS/CRYPTO, (c) EXPLORE_V2 gate strictness (tighter for conservative). |

---

## 2. **Horizon: Short-term vs Long-term**

| Idea | What it does |
|------|----------------|
| **Short horizon** | Favor recent momentum (5d, 20d), tighter TP, swing/breakout regimes. Good for active trading. |
| **Long horizon** | Favor 20d/60d trend, stronger trend filter (e.g. above 200 SMA), lower sensitivity to daily noise. Good for position holding. |
| **Implementation** | You already have `short` and `long` horizons. Add horizon-specific scoring: for `long`, boost score when 60d momentum and weekly trend align; for `short`, boost when 5d momentum and breakout/strong_bull. Optionally different momentum weights in MomentumRanker per horizon (e.g. long: 0.2/0.4/0.4 for 5d/20d/60d, short: 0.5/0.3/0.2). |

---

## 3. **Style: Momentum vs Value vs Mean-reversion vs Breakout**

| Style | Focus | How to support in scan |
|-------|--------|-------------------------|
| **Momentum** | Strong recent performance | Already partly there (MomentumRanker, regime scores). Add: rank by composite momentum; filter to top N by momentum score before regime scoring. |
| **Value / Dip** | Oversold, pullbacks in uptrend | Score boost when: in weak_bull or range + RSI low (e.g. &lt; 40) or price &lt; 20 SMA. Optional "dip buyer" profile. |
| **Mean-reversion** | Range regime, bounce from support | Boost score in RANGE regime when price near lower Bollinger or recent low. |
| **Breakout** | New highs, volume surge | Already have BREAKOUT regime. Add: volume ratio &gt; 1.5 and price at N-day high as extra boost. |

Implementation: `RECO_STYLE=momentum \| value \| mean_reversion \| breakout \| balanced`. Each style adjusts which regime/pattern bonuses apply and which filters are used.

---

## 4. **Universe construction**

| Idea | What it does |
|------|----------------|
| **Momentum-first stocks** | Already done: RECO_MOMENTUM_FILTER + min score + top N. Can extend: same for crypto (rank crypto by 5d/20d momentum, keep top N). |
| **Liquidity tiers** | For stocks: prefer "liquid" / "mega" tier; for crypto: min 24h quote volume. EXPLORE_V2 already has volume gate; make it configurable per profile. |
| **Diversification** | When building the *list* of recommendations, spread across sectors (stocks) or asset clusters (crypto). explore_v2.diversify_picks exists; use sector/cluster when returning "top 20" so you don’t get 10 tech names only. |
| **Blocklist / allowlist** | You have blocklist. Add optional allowlist: if set, only those symbols are scanned (e.g. "only these 50 stocks"). |

---

## 5. **Scoring and thresholds**

| Idea | What it does |
|------|----------------|
| **Profile-based buy threshold** | Conservative: RECO_BUY_THRESHOLD_STOCKS=65, CRYPTO=45. Aggressive: 55 and 35. So "buy" signal strength depends on profile. |
| **Eligibility min score** | Intelligence layer currently uses a fixed 40 minimum. Make it configurable (e.g. 45 conservative, 40 balanced, 35 aggressive) so more/fewer symbols become "eligible". |
| **Strong buy tier** | Already have "strong buy" (buy_thresh + 20). Add explicit "avoid" tier: score &lt; 25 or in STRONG_BEAR → don’t show as watch, or mark as "avoid". |

---

## 6. **Crypto-specific**

| Idea | What it does |
|------|----------------|
| **BTC correlation filter** | In risk-off or when BTC is dumping, reduce crypto universe or lower scores for alts (you have BTC context; use it in scoring or in universe filter). |
| **Crypto momentum filter** | Apply same idea as stocks: rank crypto by 5d/20d momentum, keep top N for scan (optional, to reduce noise). |
| **Stablecoin / low-vol exclude** | Exclude USDT/USDC-style pairs from "best opportunities" if they appear in universe. |

---

## 7. **Stocks-specific**

| Idea | What it does |
|------|----------------|
| **Sector balance** | When returning recommendations, cap per sector (e.g. max 3 per sector in top 20) so list isn’t all tech. |
| **Earnings / ex-div filter** | Already have earnings_days. Stricter: exclude or heavily penalize &lt; 7 days to earnings unless style is "event". |
| **Market cap focus** | Profile "large_cap_only" or "mid_cap": filter by market_cap_tier so scan only large/mid/small. |

---

## Suggested order to implement

1. **Scan profile (conservative / balanced / aggressive)**  
   Single knob that adjusts: min eligibility score, buy/watch thresholds, and optionally EXPLORE_V2 strictness. Fast to add, big UX win.

2. **Horizon-aware scoring**  
   Short vs long: different momentum weights or bonus rules in `generate_recommendation` so short horizon favors recent momentum, long favors trend alignment.

3. **Style (momentum / value / breakout / balanced)**  
   Second knob: which regime and pattern bonuses get extra weight. Makes the "best" list match how you want to trade.

4. **Diversification in top N**  
   When API returns "top recommendations", spread by sector (stocks) or cluster (crypto) so the list isn’t 10 of the same theme.

5. **Crypto momentum filter**  
   Optional: like stocks, rank crypto by momentum and scan only top N to focus on movers.

---

## Implemented (code)

- **Scan profile (conservative / balanced / aggressive)**  
  - Env: `RECO_PROFILE=conservative|balanced|aggressive` (default `balanced`).  
  - Drives: min eligibility score in intelligence (45 / 40 / 35), and buy/watch thresholds in the API (stocks: 65/60/52, crypto: 45/38/32, watch: 45/40/35).  
  - Explicit env `RECO_BUY_THRESHOLD_STOCKS`, `RECO_BUY_THRESHOLD_CRYPTO`, `RECO_WATCH_THRESHOLD` still override when set.

- **Horizon-aware scoring**  
  - **Short horizon**: +3 when regime is BREAKOUT/STRONG_BULL/BULL and 5d momentum > 1%.  
  - **Long horizon**: +3 when price above 200 SMA (daily) and above weekly EMA50.  
  - Applied inside `generate_recommendation(context, horizon)`.

- **RECO_STYLE**  
  - Env: `RECO_STYLE=momentum|value|mean_reversion|breakout|balanced` (default `balanced`).  
  - In `generate_recommendation`: momentum = +4 when strong short-term momentum in bull regime; value = +4 when oversold/dip in range or weak bull; mean_reversion = +3 when range + RSI ≤ 40; breakout = +4 when BREAKOUT regime.

- **Diversification in top N**  
  - When returning the recommendation list (Explore), items are spread by `diversify_key`: stocks by sector, crypto as one cluster. Uses `explore_v2.diversify_picks` when EXPLORE_V2 is enabled so the top of the list isn’t all one sector.

- **Crypto momentum filter**  
  - Env: `RECO_CRYPTO_MOMENTUM_FILTER=1` (default 0), `RECO_CRYPTO_MOMENTUM_TOP_N=80`.  
  - When enabled, crypto universe is ranked by 5d/20d/60d momentum and only the top N are scanned (same idea as stock momentum filter).
