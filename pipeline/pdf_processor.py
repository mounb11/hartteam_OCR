"""
pipeline/pdf_processor.py
--------------------------
Stap 1 van de pipeline: PDF inlezen en tekst eruit halen.

Strategie:
  1. Probeer eerst embedded tekst te lezen met pdfplumber.
     Veel PDFs (bijv. gegenereerd vanuit een EPD of Word) bevatten al digitale tekst.
     Dit is sneller en nauwkeuriger dan OCR.
  1b. Als embedded tekst vooral uit PDF "(cid:…)" font-codes bestaat (subset-fonts),
       gebruiken we OCR.
  1c. Sommige PDF's hebben kapotte font-/ToUnicode-maps: veel stuurtekens (\x01–\x1f)
       of 'versleutelde' hoofdletters zonder normale Nederlandse letter-statistieken.
       Dat is dezelfde rommel die je ziet bij kopiëren/plakken uit de viewer — dan OCR.
  2. Als een pagina geen (bruikbare) tekst bevat, is het waarschijnlijk een scan.
     Dan converteren we die pagina naar een afbeelding en geven we die door aan de OCR module.

Pagina's worden samengevoegd met alleen het geconfigureerde PAGE_JOIN_SEPARATOR (standaard formfeed \\f),
zonder leesbare koppen — zo blijft de platte tekst zo dicht mogelijk bij de bron.
"""

from pathlib import Path
import re

import pdfplumber
from loguru import logger

from pipeline.ocr import run_ocr_on_page_image
from config import PDF_RENDER_DPI, PAGE_JOIN_SEPARATOR

# Minimaal aantal tekens op een pagina om te beschouwen als "heeft tekst".
# Pagina's met minder tekens dan dit worden behandeld als scan.
MIN_CHARS_FOR_EMBEDDED_TEXT = 50

# Sommige PDF's gebruiken subset-fonts: pdfplumber geeft dan "(cid:123)" i.p.v. Unicode.
# Dat lijkt op veel tekst maar is onbruikbaar voor het LLM — dan OCR forceren.
_CID_PATTERN = re.compile(r"\(cid:\d+\)", re.IGNORECASE)


def _trim_trailing_page_noise(s: str) -> str:
    """Alleen OCR-randruis (lege regels aan einde); geen inhoud van de pagina verwijderen."""
    return s.rstrip("\n\r\f\v")


def _embedded_text_is_cid_garbage(text: str) -> bool:
    """
    True als de 'embedded' tekst vooral uit PDF character-ID placeholders bestaat.
    In dat geval moeten we OCR gebruiken om echte letters te krijgen.
    """
    if not text or "(cid" not in text:
        return False
    stripped = text.strip()
    if not stripped:
        return False
    cid_count = len(_CID_PATTERN.findall(text))
    # Na een paar (cid:…) tokens is het vrijwel zeker subset-font rommel
    if cid_count >= 12:
        return True
    # Of: (cid:-sequenties domineren het scherm
    ratio = cid_count / max(len(stripped), 1)
    return ratio >= 0.02


def _embedded_text_is_broken_glyph_map(text: str) -> bool:
    """
    True bij kapotte cmap/ToUnicode: zelfde symptoom als gekke plaktekst uit Adobe/Edge.

    Signatuur: veel ASCII-stuurtekens (behalve tab/newline), of extreem lage klinker-
    en kleine-letterverhouding voor lange blokken Latijn (geen echte Nederlandse tekst).
    """
    t = text.strip()
    if len(t) < 200:
        return False

    ctrl = sum(1 for c in t if ord(c) < 32 and c not in "\t\n\r")
    if ctrl >= max(15, len(t) // 200):
        return True
    if ctrl >= 8 and len(t) > 600:
        return True

    letters = "".join(c for c in t if c.isalpha())
    if len(letters) < 120:
        return False

    vowels = sum(1 for c in letters.lower() if c in "aeiouy")
    v_ratio = vowels / len(letters)
    lower_ratio = sum(1 for c in letters if c.islower()) / len(letters)

    if v_ratio < 0.22 and lower_ratio < 0.10:
        return True
    if v_ratio < 0.26 and lower_ratio < 0.02 and len(letters) > 400:
        return True
    return False


def extract_text_from_pdf(pdf_path: Path) -> dict:
    """
    Lees een PDF in en extraheer alle tekst, pagina voor pagina.

    Geeft terug:
        {
            "path": str,               # pad naar de PDF
            "pages": [                 # lijst van pagina's
                {
                    "page_number": int,
                    "text": str,       # geëxtraheerde tekst
                    "method": str      # "embedded" of "ocr"
                },
                ...
            ],
            "full_text": str           # alle pagina's samengevoegd
        }
    """
    logger.info(f"Verwerken gestart: {pdf_path.name}")

    result = {
        "path": str(pdf_path),
        "pages": [],
        "full_text": "",
    }

    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):

            # Stap 1: probeer embedded tekst te lezen (onvertroebeld opslaan)
            embedded_text = page.extract_text()
            if embedded_text is None:
                embedded_text = ""

            long_enough = len(embedded_text.strip()) >= MIN_CHARS_FOR_EMBEDDED_TEXT
            use_embedded = long_enough and not _embedded_text_is_cid_garbage(
                embedded_text
            ) and not _embedded_text_is_broken_glyph_map(embedded_text)

            if use_embedded:
                # Genoeg echte embedded tekst — geen OCR nodig
                logger.debug(f"  Pagina {page_number}: embedded tekst ({len(embedded_text)} tekens)")
                page_data = {
                    "page_number": page_number,
                    "text": embedded_text,
                    "method": "embedded",
                }
            else:
                if _embedded_text_is_cid_garbage(embedded_text):
                    logger.info(
                        f"  Pagina {page_number}: embedded tekst is PDF font-codes ((cid:…)), OCR wordt gebruikt"
                    )
                elif long_enough and _embedded_text_is_broken_glyph_map(embedded_text):
                    logger.info(
                        f"  Pagina {page_number}: embedded tekst heeft kapotte font-mapping "
                        f"(zoals bij kopiëren uit viewer) — OCR wordt gebruikt"
                    )
                else:
                    # Te weinig tekst — waarschijnlijk een scan
                    logger.debug(
                        f"  Pagina {page_number}: te weinig embedded tekst, OCR wordt gebruikt"
                    )
                page_image = _render_page_to_image(page)
                ocr_text = run_ocr_on_page_image(page_image)
                page_data = {
                    "page_number": page_number,
                    "text": _trim_trailing_page_noise(ocr_text),
                    "method": "ocr",
                }

            result["pages"].append(page_data)

    # Geen leesbare paginakoppen — alleen scheidingsteken (standaard formfeed)
    result["full_text"] = PAGE_JOIN_SEPARATOR.join(p["text"] for p in result["pages"])

    logger.success(f"  Klaar: {len(result['pages'])} pagina('s) verwerkt")
    return result


def _render_page_to_image(page):
    """
    Rendert één PDF-pagina als een PIL-afbeelding op hoge resolutie.
    Wordt alleen aangeroepen als OCR nodig is.

    PyMuPDF (fitz) bundled libraries — geen aparte Poppler-installatie (anders wel
    vereist door pdf2image op Windows).
    """
    import fitz
    from PIL import Image

    pdf_path = page.pdf.stream.name
    page_index = page.page_number - 1
    scale = PDF_RENDER_DPI / 72.0
    matrix = fitz.Matrix(scale, scale)

    doc = fitz.open(pdf_path)
    try:
        pdf_page = doc.load_page(page_index)
        pix = pdf_page.get_pixmap(matrix=matrix, alpha=False)
        return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    finally:
        doc.close()
