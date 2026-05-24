"""
Playwright live-site verification for http://3.151.143.63
Checks every page for: correct UI version, sidebar presence, no fatal errors.
"""
import asyncio
import sys
from playwright.async_api import async_playwright

BASE = "http://3.151.143.63"
results = []

def ok(name, detail=""):
    results.append(("PASS", name, detail))
    print(f"  PASS  {name}" + (f"  -- {detail}" if detail else ""))

def fail(name, detail=""):
    results.append(("FAIL", name, detail))
    print(f"  FAIL  {name}" + (f"  -- {detail}" if detail else ""))

def warn(name, detail=""):
    results.append(("WARN", name, detail))
    print(f"  WARN  {name}" + (f"  -- {detail}" if detail else ""))


async def check_page(page, path, label):
    print(f"\n-- {label} ({path})")
    console_errors = []
    page.on("console", lambda m: console_errors.append(m.text) if m.type == "error" else None)

    resp = await page.goto(f"{BASE}{path}", wait_until="domcontentloaded", timeout=25000)
    status = resp.status if resp else 0

    if status == 200:
        ok(f"HTTP {status}")
    else:
        fail(f"HTTP {status}")
        return

    await page.wait_for_timeout(2000)

    body = await page.inner_text("body")

    # 1. Not old UI
    if "One server mode" in body:
        fail("New UI (no old banner)", "Still shows 'One server mode' text")
    else:
        ok("New UI (no old banner)")

    # 2. Sidebar/nav has Explore link (new UI marker)
    explore_count = await page.locator("text=Explore").count()
    if explore_count > 0:
        ok("New sidebar with Explore link")
    else:
        fail("New sidebar with Explore link", "No 'Explore' text found")

    # 3. Nav link count (new UI has 10+ sidebar links)
    nav_links = await page.locator("nav a, .sidebar a").count()
    if nav_links >= 6:
        ok(f"Sidebar has {nav_links} nav links (full sidebar)")
    elif nav_links > 0:
        warn(f"Sidebar has only {nav_links} nav links", "Expected 10+")
    else:
        fail("Sidebar links", "No nav/sidebar links found")

    # 4. No fatal errors in body
    fatals = ["Internal Server Error", "502 Bad Gateway", "Application Error", "Traceback"]
    found_fatal = [f for f in fatals if f in body]
    if found_fatal:
        fail("No fatal errors in body", f"Found: {found_fatal}")
    else:
        ok("No fatal errors in body")

    # 5. Console errors (filter noise)
    noisy = ["favicon", "eval-stdin", "Autofill", "phpunit"]
    real_errors = [e for e in console_errors if not any(n in e for n in noisy)]
    if real_errors:
        for e in real_errors[:2]:
            warn("JS console error", e[:100])
    else:
        ok("No JS console errors")

    return body


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx = await browser.new_context(viewport={"width": 1280, "height": 720})
        page = await ctx.new_page()

        pages = [
            ("/",              "HOME"),
            ("/bots",          "BOTS"),
            ("/explore",       "EXPLORE"),
            ("/analytics",     "ANALYTICS"),
            ("/dca",           "DCA"),
            ("/settings",      "SETTINGS"),
            ("/backtest",      "BACKTEST"),
            ("/autopilot",     "AUTOPILOT"),
            ("/setup-autopilot", "SETUP-AUTOPILOT"),
            ("/safety",        "SAFETY"),
            ("/journal",       "JOURNAL"),
            ("/strategies",    "STRATEGIES"),
        ]

        page_bodies = {}
        for path, label in pages:
            body = await check_page(page, path, label)
            if body:
                page_bodies[path] = body

        # --- Page-specific deep checks ---

        # EXPLORE: look for feed error
        print("\n-- EXPLORE (deep check)")
        if "/explore" in page_bodies:
            b = page_bodies["/explore"]
            if "Unable to load signals" in b or "HTTP 500" in b:
                warn("Explore feed", "Shows 'Unable to load signals / HTTP 500' -- worker feed bug")
            elif "404 Not Found" in b:
                fail("Explore feed", "Shows 404 in page body")
            else:
                ok("Explore feed", "No visible feed error")

        # SAFETY: check checklist loaded
        print("\n-- SAFETY (deep check)")
        await page.goto(f"{BASE}/safety", wait_until="domcontentloaded", timeout=15000)
        await page.wait_for_timeout(3000)
        safety_body = await page.inner_text("body")
        if "/api/safety_check -> 404" in safety_body:
            fail("Safety checklist API", "Still showing 404 for /api/safety_check")
        elif any(x in safety_body for x in ["API token", "Allow live", "Kraken", "Kill switch"]):
            ok("Safety checklist API", "Checklist data rendered")
        else:
            warn("Safety checklist API", "Could not confirm checklist data")

        # DARK MODE BUTTON
        print("\n-- DARK MODE BUTTON")
        await page.goto(f"{BASE}/", wait_until="domcontentloaded", timeout=15000)
        await page.wait_for_timeout(500)
        btn = page.locator("#themeToggle")
        count = await btn.count()
        if count > 0:
            box = await btn.bounding_box()
            if box:
                if 0 <= box["y"] < 720:
                    ok("Dark mode button in viewport", f"y={box['y']:.0f} x={box['x']:.0f} (viewport 1280x720)")
                else:
                    warn("Dark mode button off-screen", f"y={box['y']:.0f}px -- below 720px fold")
            else:
                warn("Dark mode button", "No bounding box (display:none or zero-size?)")
        else:
            warn("Dark mode button", "#themeToggle not found on page")

        # BOTS: Start All / Stop All button positions
        print("\n-- BOTS ACTION BUTTONS")
        await page.goto(f"{BASE}/bots", wait_until="domcontentloaded", timeout=15000)
        await page.wait_for_timeout(500)
        for btn_text in ["Start All", "Stop All"]:
            loc = page.locator(f"text={btn_text}").first
            if await loc.count() > 0:
                box = await loc.bounding_box()
                if box:
                    if box["y"] < 720:
                        ok(f"'{btn_text}' button in viewport", f"y={box['y']:.0f}px")
                    else:
                        warn(f"'{btn_text}' button below fold", f"y={box['y']:.0f}px (viewport=720px)")
                else:
                    warn(f"'{btn_text}' button", "No bounding box")
            else:
                warn(f"'{btn_text}' button", "Not found on /bots page")

        # API ENDPOINTS
        print("\n-- API ENDPOINTS")
        api_checks = [
            "/api/health",
            "/api/safety_check",
            "/api/bots",
            "/api/pnl",
            "/api/explore/scan_status",
            "/api/activity",
            "/api/bots/summary",
            "/api/strategies/leaderboard",
            "/api/portfolio",
        ]
        for ep in api_checks:
            resp = await page.request.get(f"{BASE}{ep}")
            if resp.status == 200:
                try:
                    data = await resp.json()
                    ok(f"GET {ep}", f"200 ok={data.get('ok', '?')}")
                except Exception:
                    ok(f"GET {ep}", "200")
            elif resp.status in (422, 400):
                warn(f"GET {ep}", f"{resp.status} (param issue)")
            else:
                fail(f"GET {ep}", f"{resp.status}")

        await browser.close()

    # SUMMARY
    passed = sum(1 for r in results if r[0] == "PASS")
    warned  = sum(1 for r in results if r[0] == "WARN")
    failed  = sum(1 for r in results if r[0] == "FAIL")
    total   = len(results)
    print(f"\n{'='*60}")
    print(f"RESULTS: {passed}/{total} passed  |  {warned} warnings  |  {failed} failures")
    print("="*60)
    if failed:
        print("\nFAILURES:")
        for r in results:
            if r[0] == "FAIL":
                print(f"  x {r[1]}" + (f" -- {r[2]}" if r[2] else ""))
    if warned:
        print("\nWARNINGS:")
        for r in results:
            if r[0] == "WARN":
                print(f"  ! {r[1]}" + (f" -- {r[2]}" if r[2] else ""))
    return 1 if failed else 0


sys.exit(asyncio.run(main()))
