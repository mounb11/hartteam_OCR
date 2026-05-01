"""
pipeline/validator.py
---------------------
Stap 3: Validatie van de geëxtraheerde data.

  - Verplichte / aanbevolen velden
  - BSN via 11-proef
  - Datumformaten
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from loguru import logger


@dataclass
class ValidationResult:
    """Resultaat van de validatiestap."""

    is_valid: bool = True
    needs_human_review: bool = False
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add_error(self, message: str):
        self.errors.append(message)
        self.is_valid = False
        self.needs_human_review = True

    def add_warning(self, message: str):
        self.warnings.append(message)
        self.needs_human_review = True


def validate_extracted_data(extracted: dict) -> ValidationResult:
    """
    Valideer de geëxtraheerde patientdata.

    Parameters:
        extracted: dict — output van extractor.extract_patient_data()

    Returns:
        ValidationResult met fouten en waarschuwingen.
    """
    result = ValidationResult()

    if not extracted.get("extraction_success", False):
        error_msg = extracted.get("extraction_error", "Onbekende extractiefout")
        result.add_error(f"Extractie mislukt: {error_msg}")
        return result

    patient = extracted.get("patient", {}) or {}
    medisch = extracted.get("medisch", {}) or {}

    bsn = patient.get("bsn")
    if bsn:
        bsn_error = _validate_bsn(bsn)
        if bsn_error:
            result.add_error(f"BSN ongeldig: {bsn_error}")
    else:
        result.add_warning("BSN ontbreekt — beperkte koppelbaarheid tot patiëntrecord")

    if not patient.get("achternaam"):
        result.add_warning("Achternaam patiënt ontbreekt")

    if not patient.get("geboortedatum"):
        result.add_warning("Geboortedatum patiënt ontbreekt")
    else:
        date_error = _validate_date(patient["geboortedatum"])
        if date_error:
            result.add_error(f"Geboortedatum ongeldig: {date_error}")

    if not medisch.get("hoofddiagnose"):
        result.add_warning("Hoofddiagnose ontbreekt")

    doc_datum = extracted.get("document", {}).get("datum")
    if doc_datum:
        date_error = _validate_date(doc_datum)
        if date_error:
            result.add_error(f"Documentdatum ongeldig: {date_error}")

    if result.errors:
        logger.warning(f"  Validatie: {len(result.errors)} fout(en), {len(result.warnings)} waarschuwing(en)")
    elif result.warnings:
        logger.warning(f"  Validatie: 0 fouten, {len(result.warnings)} waarschuwing(en) — review aanbevolen")
    else:
        logger.success("  Validatie: geslaagd, geen problemen gevonden")

    return result


def _validate_bsn(bsn: str) -> str | None:
    """Valideer een BSN via de officiële 11-proef."""
    bsn_clean = bsn.replace(" ", "").replace("-", "")

    if not bsn_clean.isdigit():
        return f"BSN bevat niet-cijfer tekens: '{bsn}'"

    if len(bsn_clean) != 9:
        return f"BSN heeft {len(bsn_clean)} cijfers (verwacht: 9)"

    digits = [int(d) for d in bsn_clean]
    gewichten = [9, 8, 7, 6, 5, 4, 3, 2, -1]
    som = sum(d * g for d, g in zip(digits, gewichten))

    if som % 11 != 0:
        return f"BSN '{bsn_clean}' faalt de 11-proef"

    return None


def _validate_date(date_str: str) -> str | None:
    try:
        datetime.strptime(date_str, "%d-%m-%Y")
        return None
    except ValueError:
        return f"'{date_str}' is geen geldige datum (verwacht formaat: DD-MM-YYYY)"
