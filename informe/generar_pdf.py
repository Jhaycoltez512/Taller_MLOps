"""Convierte informe/informe.html en informe/informe.pdf con Chromium (Playwright).

Uso: python informe/generar_pdf.py [--chromium /ruta/a/chrome]
Requiere el paquete playwright y un Chromium instalado.
"""
import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright

ap = argparse.ArgumentParser()
ap.add_argument("--chromium", default=None)
args = ap.parse_args()
html = Path("informe/informe.html").resolve()
with sync_playwright() as p:
    browser = p.chromium.launch(executable_path=args.chromium)
    page = browser.new_page()
    page.goto(html.as_uri())
    page.wait_for_load_state("networkidle")
    page.pdf(path="informe/informe.pdf", format="A4", print_background=True, prefer_css_page_size=True)
    browser.close()
print("informe/informe.pdf")
