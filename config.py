"""
config.py
---------
  - Tesseract voor OCR
  - Ollama (qwen2.5:14b) voor semantische extractie
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Laad omgevingsvariabelen uit .env bestand (staat NIET in git)
load_dotenv()

# ── Mappenstructuur ────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent

# Map waar inkomende PDFs worden neergezet
INPUT_DIR = BASE_DIR / "input"

# Map waar verwerkte output (JSON) wordt opgeslagen
OUTPUT_DIR = BASE_DIR / "output"

# Map voor logs
LOG_DIR = BASE_DIR / "logs"

# Testdata
TEST_PDF_DIR = BASE_DIR / "tests" / "test_pdfs"

# Maak mappen aan als ze nog niet bestaan
for d in [INPUT_DIR, OUTPUT_DIR, LOG_DIR]:
    d.mkdir(exist_ok=True)

# ── Tesseract OCR ──────────────────────────────────────────────────────────────
# Op macOS via Homebrew: /opt/homebrew/bin/tesseract
# Op Linux: /usr/bin/tesseract
TESSERACT_CMD = os.getenv("TESSERACT_CMD", "/opt/homebrew/bin/tesseract")
TESSERACT_LANGUAGES = "nld+eng"
# DPI voor het renderen van PDF-pagina's naar afbeelding voor OCR
# 300 DPI is standaard voor goede OCR kwaliteit
PDF_RENDER_DPI = 300
# OCR: alleen bij > 1.0 contrast verhogen (anders geen inhoudelijke beeldwijziging)
OCR_CONTRAST_FACTOR = float(os.getenv("OCR_CONTRAST_FACTOR", "1.0"))
# Tesseract paginamodus (3 = volledige auto zonder oriëntatiedetectie; 6 = één tekstblok)
TESSERACT_PSM = int(os.getenv("TESSERACT_PSM", "3"))

# Bij samenvoegen van paginatekst géén leesbare labels — alleen ASCII formfeed (gangbare PDF-tekstvorm)
PAGE_JOIN_SEPARATOR = os.getenv("PAGE_JOIN_SEPARATOR", "\f")

# Schrijf volledige extractie ook als platte tekst naast JSON
SAVE_VERBATIM_TEXT_FILE = os.getenv("SAVE_VERBATIM_TEXT_FILE", "1").strip().lower() in (
    "1",
    "true",
    "yes",
)

# ── Ollama (lokale LLM voor semantische extractie) ────────────────────────────#
# Model: qwen2.5:14b
#   - 9 GB op schijf (Q4_K_M quantisatie)
#   - Installeren: ollama pull qwen2.5:14b
# Ollama basis-URL — standaard lokaal op poort 11434
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
# Het model dat Ollama gebruikt voor extractie
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:14b")
# Maximaal aantal tokens dat het model mag genereren als antwoord
OLLAMA_MAX_TOKENS = int(os.getenv("OLLAMA_MAX_TOKENS", "8192"))

# Max. tekstlengte naar het LLM; 0 = geen truncatie van brondocument voor het model
OLLAMA_DOCUMENT_MAX_CHARS = int(os.getenv("OLLAMA_DOCUMENT_MAX_CHARS", "0"))

# ── HiX / ChipSoft API ─────────────────────────────────────────────────────────

# Base URL van de HiX FHIR endpoint (on-premise HiX installatie)
HIX_FHIR_BASE_URL = os.getenv("HIX_FHIR_BASE_URL", "http://hix-server/fhir/R4")
# Authenticatie voor HiX API (Bearer token of Basic auth)
HIX_API_TOKEN = os.getenv("HIX_API_TOKEN")
# Timeout in seconden voor API calls naar HiX
HIX_TIMEOUT_SECONDS = 30

# ── Pipeline gedrag ────────────────────────────────────────────────────────────

# Of de pipeline stopt bij een fout, of doorgaat met de volgende PDF
STOP_ON_ERROR = False
