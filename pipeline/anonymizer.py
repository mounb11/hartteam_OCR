"""
pipeline/anonymizer.py
----------------------
Verwijder persoonsgegevens uit geëxtraheerde data en ruwe OCR-tekst.

Verwijdert:
  - Patiëntnaam (achternaam, voornaam), geboortedatum, BSN, adres, postcode, woonplaats
  - Namen van behandelend en verwijzend arts
  - Telefoonnummers en e-mailadressen
  - T.a.v.-regels (huisarts, PA, specialist als ontvanger)
  - Straatnamen + huisnummers
  - Nederlandse postcodes + plaatsnamen
  - Patiëntnummers en BSN-nummers

Geslacht, ziekenhuis- en afdelingsnamen worden bewaard (niet-persoonsgebonden).
"""

from __future__ import annotations

import copy
import re
from typing import Any

# ── Regex patronen ────────────────────────────────────────────────────────────

_PHONE_RE = re.compile(
    r'\b(?:\+31|0031|0)[\s.\-]?\(?\d{1,4}\)?[\s.\-]?\d[\s.\-]?\d[\s.\-]?\d[\s.\-]?\d[\s.\-]?\d[\s.\-]?\d[\s.\-]?\d?\b'
)
_EMAIL_RE = re.compile(
    r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b'
)
_PATIENTNR_RE = re.compile(
    r'(?:Pati[eë]ntn(?:r\.?|ummer)|Pat\.?\s*nr\.?(?:/BSN)?|Patnr\.?)\s*[/:\.]?\s*[^\n]+',
    re.IGNORECASE,
)
_BSN_RE = re.compile(
    r'\bBSN\s*[:/]?\s*\d{6,9}\b',
    re.IGNORECASE,
)
# "T.a.v. mevr. Dorresteijn huisarts" → volledige regel
_TAV_RE = re.compile(
    r'T\.?\s*a\.?\s*v\.[\s.]*[^\n]+',
    re.IGNORECASE,
)
# Nederlandse postcode (bijv. "2314 ZB" of "2314ZB"), optioneel gevolgd door plaatsnaam
_POSTCODE_RE = re.compile(
    r'\b\d{4}\s?[A-Z]{2}(?:\s+[A-Z][a-zA-Z\-]+(?:\s+[a-zA-Z\-]+)*)?\b'
)
# Straatnaam + huisnummer op basis van gangbare Nederlandse straattype-suffixen
_STRAAT_RE = re.compile(
    r'\b\w+(?:straat|weg|laan|plein|kade|gracht|singel|dijk|dam|ring|dreef|pad|steeg|hofje|poort|markt|boulevard|allee|baan|zijde|veld|hoek)\s+\d+[a-zA-Z]{0,2}\b',
    re.IGNORECASE,
)

# ── PII-velddefinities ────────────────────────────────────────────────────────

_PATIENT_PII = ('achternaam', 'voornaam', 'geboortedatum', 'bsn', 'adres', 'postcode', 'woonplaats')
_MEDISCH_PII = ('behandelend_arts', 'verwijzend_arts')
_NAME_FIELDS  = ('achternaam', 'voornaam', 'behandelend_arts', 'verwijzend_arts')


def _collect_pii_values(extracted: dict) -> list[str]:
    """
    Verzamel niet-null PII-strings uit de geëxtraheerde data.

    Voor naamvelden worden ook losse woorden toegevoegd (bijv. 'Jansen' naast
    'Dr. Jan Jansen'), zodat gedeeltelijke matches in vrije tekst ook worden
    vervangen. Resultaat gesorteerd op aflopende lengte (langste eerst).
    """
    raw: list[str] = []

    patient = extracted.get('patient') or {}
    for field in _PATIENT_PII:
        v = patient.get(field)
        if v and isinstance(v, str):
            v = v.strip()
            if v:
                raw.append(v)
                if field in _NAME_FIELDS:
                    for part in v.split():
                        part = part.strip('.,;:()')
                        if len(part) > 2:
                            raw.append(part)

    medisch = extracted.get('medisch') or {}
    for field in _MEDISCH_PII:
        v = medisch.get(field)
        if v and isinstance(v, str):
            v = v.strip()
            if v:
                raw.append(v)
                for part in v.split():
                    part = part.strip('.,;:()')
                    if len(part) > 2:
                        raw.append(part)

    seen: set[str] = set()
    result: list[str] = []
    for v in sorted(raw, key=len, reverse=True):
        key = v.lower()
        if key not in seen:
            seen.add(key)
            result.append(v)

    return result


def _apply_regexes(text: str) -> str:
    """Pas alle regex-patronen toe op een string."""
    text = _PATIENTNR_RE.sub('[GEANONIMISEERD]', text)
    text = _BSN_RE.sub('[GEANONIMISEERD]', text)
    text = _TAV_RE.sub('[GEANONIMISEERD]', text)
    text = _STRAAT_RE.sub('[GEANONIMISEERD]', text)
    text = _POSTCODE_RE.sub('[GEANONIMISEERD]', text)
    text = _PHONE_RE.sub('[TEL]', text)
    text = _EMAIL_RE.sub('[EMAIL]', text)
    return text


def _scrub_strings(value: Any) -> Any:
    """Pas alle regex-patronen recursief toe op alle strings in een structuur."""
    if isinstance(value, str):
        return _apply_regexes(value)
    if isinstance(value, list):
        return [_scrub_strings(v) for v in value]
    if isinstance(value, dict):
        return {k: _scrub_strings(v) for k, v in value.items()}
    return value


def anonymize_document_text(text: str, extracted: dict) -> str:
    """
    Anonimiseer ruwe OCR-tekst.

    Stap 1: vervang bekende PII-waarden (uit de extractie) door '[GEANONIMISEERD]'.
    Stap 2: regex-sweep (straat, postcode, T.a.v., patiëntnr, BSN, tel, email).
    """
    if not text:
        return text

    pii_values = _collect_pii_values(extracted)
    for value in pii_values:
        pattern = re.compile(r'\b' + re.escape(value) + r'\b', re.IGNORECASE)
        text = pattern.sub('[GEANONIMISEERD]', text)

    return _apply_regexes(text)


def anonymize_extracted(extracted: dict) -> dict:
    """
    Geeft een geanonimiseerde deep copy van de geëxtraheerde data.

    Stap 1: bekende PII-velden op null zetten.
    Stap 2: regex-sweep over alle overige strings.
    """
    data = copy.deepcopy(extracted)

    if isinstance(data.get('patient'), dict):
        for field in _PATIENT_PII:
            if field in data['patient']:
                data['patient'][field] = None

    if isinstance(data.get('medisch'), dict):
        for field in _MEDISCH_PII:
            if field in data['medisch']:
                data['medisch'][field] = None

    return _scrub_strings(data)
