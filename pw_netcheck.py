"""Find exact URLs causing 404/502 on each page via network events."""
import asyncio
from playwright.async_api import async_playwright

BASE = "http://3.151.143.63"
PAGES = ["/", "/bots", "/explore", "/analytics", "/dca",
         "/safety", "/journal", "/strategies"]

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx = await browser.new_context(viewport={"width": 1280, "height": 720})
        page = await ctx.new_page()

        for path in PAGES:
            failures = []

            def on_response(resp):
                if resp.status in (404, 502, 500) and "/api/" in resp.url:
                    failures.append((resp.status, resp.url.replace(BASE, "")))

            page.on("response", on_response)
            await page.goto(f"{BASE}{path}", wait_until="domcontentloaded", timeout=20000)
            await page.wait_for_timeout(3000)
            page.remove_listener("response", on_response)

            if failures:
                print(f"\n{path}:")
                for status, url in sorted(set(failures)):
                    print(f"  {status}  {url}")

        await browser.close()

asyncio.run(main())
