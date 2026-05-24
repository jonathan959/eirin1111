"""Full Playwright audit: visit every page, capture EVERY failed network request and console error."""
import asyncio
from collections import defaultdict
from playwright.async_api import async_playwright

BASE = "http://3.151.143.63"
PAGES = [
    "/", "/bots", "/explore", "/analytics", "/dca", "/settings",
    "/backtest", "/autopilot", "/setup-autopilot", "/safety",
    "/journal", "/strategies", "/dashboard",
]

async def audit_page(browser, path):
    page = await browser.new_page()
    failures = []  # (url, status, method)
    console_errs = []

    def on_response(resp):
        if resp.status >= 400:
            failures.append((resp.url, resp.status, resp.request.method))

    def on_console(msg):
        if msg.type == "error":
            console_errs.append(msg.text[:300])

    def on_pageerror(err):
        console_errs.append(f"PAGEERROR: {str(err)[:300]}")

    page.on("response", on_response)
    page.on("console", on_console)
    page.on("pageerror", on_pageerror)

    try:
        await page.goto(f"{BASE}{path}", wait_until="load", timeout=90000)
        # Wait for late XHRs
        await page.wait_for_timeout(8000)
    except Exception as e:
        console_errs.append(f"NAV_ERROR: {str(e)[:200]}")

    await page.close()
    return failures, console_errs

async def main():
    all_failures = defaultdict(list)  # path -> list of (url, status, method)
    all_errors = defaultdict(list)    # path -> console errors

    async with async_playwright() as pw:
        browser = await pw.chromium.launch()
        for path in PAGES:
            print(f"\n[{path}] auditing...", flush=True)
            f, e = await audit_page(browser, path)
            all_failures[path] = f
            all_errors[path] = e
            for url, status, method in f:
                # Strip query string for grouping
                short = url.replace(BASE, "").split("?")[0]
                print(f"  FAIL {status} {method} {short}")
            for err in e:
                print(f"  ERR  {err[:200]}")
        await browser.close()

    # Aggregate summary: unique failing endpoints
    print("\n" + "=" * 80)
    print("AGGREGATE FAILING ENDPOINTS (across all pages):")
    print("=" * 80)
    seen = {}
    for path, fails in all_failures.items():
        for url, status, method in fails:
            short = url.replace(BASE, "").split("?")[0]
            key = (method, short, status)
            seen.setdefault(key, []).append(path)
    for (method, short, status), pages in sorted(seen.items()):
        print(f"  {status} {method:6} {short:60} <- on pages: {', '.join(pages)}")

    print("\n" + "=" * 80)
    print("AGGREGATE CONSOLE ERRORS (unique):")
    print("=" * 80)
    seen_errs = {}
    for path, errs in all_errors.items():
        for e in errs:
            # Strip http response/resource boilerplate
            key = e.split(" -- ")[-1][:150] if "Failed to load resource" not in e else "Failed to load resource"
            if "Failed to load resource" in e:
                continue  # Already covered by failures
            seen_errs.setdefault(e[:150], []).append(path)
    for err, pages in sorted(seen_errs.items()):
        print(f"  {err}")
        print(f"     on pages: {', '.join(set(pages))}")

asyncio.run(main())
