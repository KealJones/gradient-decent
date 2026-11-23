"""Headless check of filter/: runs window.selfTest() and prints the result. Needs playwright and Chrome.

    python3 -m http.server 8000 &
    python3 tools/check_filter.py [http://localhost:8000/filter/] [path/to/chrome]
"""
import json
import sys
from playwright.sync_api import sync_playwright

url = sys.argv[1] if len(sys.argv) > 1 else 'http://localhost:8000/filter/'
with sync_playwright() as p:
    opts = {'executable_path': sys.argv[2], 'args': ['--no-sandbox']} if len(sys.argv) > 2 else {'channel': 'chrome'}
    browser = p.chromium.launch(**opts)
    page = browser.new_page()
    page.on('pageerror', lambda e: print('page error:', e))
    page.goto(url)
    page.wait_for_timeout(500)
    print(page.text_content('#stat'))
    print(json.dumps(page.evaluate('selfTest()'), indent=1))
    browser.close()
