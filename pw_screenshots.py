"""Take screenshots of /bots and /explore for visual verification."""
import asyncio
from playwright.async_api import async_playwright

BASE = "http://3.151.143.63"

async def main():
    async with async_playwright() as pw:
        browser = await pw.chromium.launch()
        ctx = await browser.new_context(viewport={"width": 1440, "height": 900})
        page = await ctx.new_page()
        for path, fn in [("/", "after_dashboard.png"), ("/bots", "after_bots.png"),
                          ("/explore", "after_explore.png"), ("/safety", "after_safety.png")]:
            print(f"-> {path}")
            try:
                await page.goto(f"{BASE}{path}", wait_until="load", timeout=90000)
                await page.wait_for_timeout(10000)  # let XHRs settle
                await page.screenshot(path=fn, full_page=True)
                print(f"   saved {fn}")
            except Exception as e:
                print(f"   ERROR: {e}")
        await browser.close()

asyncio.run(main())
