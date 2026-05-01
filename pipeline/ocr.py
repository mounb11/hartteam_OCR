"""
pipeline/ocr.py
---------------
Stap 1b: OCR met Tesseract (alleen voor gescande pagina's).

Standaard géén contrastbeeld-bewerking (OCR_CONTRAST_FACTOR=1.0) om de bronpixelinformatie
niet te vervormen — alleen hogere factoren activeren lichte contrastversterking.
"""

import pytesseract
from PIL import Image, ImageEnhance
from loguru import logger

from config import (
    TESSERACT_CMD,
    TESSERACT_LANGUAGES,
    OCR_CONTRAST_FACTOR,
    TESSERACT_PSM,
)

pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD


def run_ocr_on_page_image(image: Image.Image) -> str:
    """
    Voer OCR uit op een PIL-afbeelding van een PDF-pagina.

    Bij OCR_CONTRAST_FACTOR > 1 wordt alleen dan grijswaarden + contrast toegepast;
    anders wordt de gerenderde RGB-afbeelding direct aan Tesseract gegeven (geen extra bewerking).

    `--psm` en `preserve_interword_spaces` zijn configureerbaar voor leesorde en spaties.
    """
    logger.debug("  OCR uitvoeren op pagina-afbeelding...")

    if OCR_CONTRAST_FACTOR > 1.001:
        work = _preprocess_grayscale_contrast(image, OCR_CONTRAST_FACTOR)
    else:
        work = image if image.mode == "RGB" else image.convert("RGB")

    custom_config = (
        f"--oem 3 --psm {TESSERACT_PSM} -c preserve_interword_spaces=1"
    )

    text = pytesseract.image_to_string(
        work,
        lang=TESSERACT_LANGUAGES,
        config=custom_config,
    )

    logger.debug(f"  OCR klaar: {len(text)} tekens herkend")
    return text


def _preprocess_grayscale_contrast(image: Image.Image, factor: float) -> Image.Image:
    """Optioneel: grijswaarden + contrast (alleen als factor > 1)."""
    im = image.convert("L")
    enhancer = ImageEnhance.Contrast(im)
    return enhancer.enhance(factor)
