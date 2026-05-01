"""
pipeline/extractor.py
---------------------
Stap 2: Semantische data-extractie met een lokale LLM via Ollama.
"""

from __future__ import annotations

import json
import re
from typing import Any

from loguru import logger
import ollama

from config import (
    OLLAMA_MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_MAX_TOKENS,
    OLLAMA_DOCUMENT_MAX_CHARS,
)


EXTRACTION_SCHEMA: dict[str, Any] = {
    "patient": {
        "achternaam": "string | null",
        "voornaam": "string | null",
        "geboortedatum": "string in formaat DD-MM-YYYY | null",
        "bsn": "9-cijferig burgerservicenummer als string | null",
        "geslacht": "M of V of null",
        "adres": "string | null",
        "postcode": "string | null",
        "woonplaats": "string | null",
    },
    "document": {
        "datum": "string in formaat DD-MM-YYYY | null",
        "type": "bijv. ontslagbrief / verwijsbrief / polikliniekbrief | null",
        "ziekenhuis_van_herkomst": "naam van het ziekenhuis dat het document stuurde | null",
        "afdeling": "string | null",
    },
    "medisch": {
        "hoofddiagnose": "string | null",
        "nevendiagnoses": ["lijst van strings"],
        "medicatie": [
            {
                "naam": "string",
                "dosering": "string | null",
                "frequentie": "string | null",
            }
        ],
        "allergieën": ["lijst van strings"],
        "behandelend_arts": "naam van de arts | null",
        "verwijzend_arts": "naam van de verwijzende arts | null",
        "reden_van_verwijzing": "string | null",
    },
    "extractie_metadata": {
        "ontbrekende_velden": ["lijst van velden die je niet kon vinden in de tekst"],
        "opmerkingen": "string | null — bijzonderheden over dit document",
    },
}


SYSTEM_PROMPT = """Je bent een medisch documentanalyse-assistent voor een Nederlands ziekenhuis.
Je taak is om gestructureerde patiëntinformatie te extraheren uit tekst van medische documenten.

Regels:
- Je antwoord is ALLEEN ÉÉN geldig JSON-object (UTF-8), zonder Markdown, zonder tekst eromheen.
- Alle strings tussen dubbele aanhalingstekens ". Gebruik geen apostroffen als quote voor keys.
- Gebruik null voor velden die je niet vindt — verzin nooit inhoud.
- Voor elk ingevuld tekstveld: kopieer de woorden spelling-onveranderd uit het document (verbatim), geen parafrase of vertaling.
- Geen komma na het laatste veld van een object of array (geen trailing comma).
- BSN alleen als exact 9 cijfers als string — anders null.
- Datums als DD-MM-YYYY of null.
"""


CODE_BLOCK_PATTERN = re.compile(
    r"```(?:json)?\s*(.+?)\s*```",
    re.DOTALL | re.IGNORECASE,
)


def _maybe_truncate_document_text(full_text: str) -> str:
    """
    Alleen gebruikt voor het LLM-prompt. Nul of negatief = geen verkorting van de tekstbron.
    (Eventueel ingekorten stuk wordt nergens als complete extractie beschouwd — zie pipeline JSON.)
    """
    if OLLAMA_DOCUMENT_MAX_CHARS <= 0 or len(full_text) <= OLLAMA_DOCUMENT_MAX_CHARS:
        return full_text
    logger.warning(
        f"Tekst naar LLM ingekort tot {OLLAMA_DOCUMENT_MAX_CHARS} tekens "
        "(OLLAMA_DOCUMENT_MAX_CHARS); JSON/output bevat wél het volledige document."
    )
    return full_text[: OLLAMA_DOCUMENT_MAX_CHARS]


def _extract_balanced_json_slice(text: str, start_idx: int) -> str | None:
    """Eerste complete '{'…'}' blok met respect voor quotes en escapes."""
    depth = 0
    in_string: str | None = None
    escape = False

    for i in range(start_idx, len(text)):
        c = text[i]

        if in_string is not None:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == in_string:
                in_string = None
            continue

        if c in ('"', "'"):
            in_string = c
            continue

        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start_idx : i + 1]
    return None


def _coerce_with_json_repair(raw: str) -> dict:
    """Laatste redmiddel: repareer gebroken JSON (trailing comma, quotes, enz.)."""
    from json_repair import loads as repair_loads

    out = repair_loads(raw)
    if isinstance(out, dict):
        return out
    raise TypeError(f"json_repair gaf geen object: {type(out)}")


def _parse_llm_json(text: str) -> dict:
    """Parseert modeloutput tot dict; meerdere strategieën."""
    text = (text or "").strip()
    if not text:
        raise json.JSONDecodeError("Lege modeloutput", text, 0)

    tries: list[tuple[str, str]] = [("rauw", text)]

    m = CODE_BLOCK_PATTERN.search(text)
    if m:
        inner = m.group(1).strip()
        tries.append(("code_block", inner))

    for _label, blob in tries:
        try:
            return json.loads(blob.strip())
        except json.JSONDecodeError:
            pass

        start = blob.find("{")
        if start != -1:
            balanced = _extract_balanced_json_slice(blob, start)
            if balanced:
                try:
                    return json.loads(balanced)
                except json.JSONDecodeError:
                    try:
                        return _coerce_with_json_repair(balanced)
                    except Exception:
                        pass

    start = text.find("{")
    if start != -1:
        balanced = _extract_balanced_json_slice(text, start)
        if balanced:
            try:
                return json.loads(balanced)
            except json.JSONDecodeError:
                pass
            try:
                return _coerce_with_json_repair(balanced)
            except Exception:
                pass

    try:
        return _coerce_with_json_repair(text)
    except Exception as e:
        raise json.JSONDecodeError(f"json_repair faalde: {e}", text, 0) from e


def _chat_extract(
    client: ollama.Client,
    *,
    messages: list[dict[str, str]],
    force_json_format: bool,
) -> str:
    kwargs: dict[str, Any] = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "options": {
            "num_predict": OLLAMA_MAX_TOKENS,
            "temperature": 0.05,
            "top_p": 0.85,
        },
    }
    if force_json_format:
        kwargs["format"] = "json"

    response = client.chat(**kwargs)
    return (response.get("message") or {}).get("content") or ""


def _repair_json_via_llm(client: ollama.Client, broken: str, err: str) -> dict:
    """Tweede ronde: laat het model kapotte JSON corrigeren (format=json)."""
    clipped = broken.strip()[:20000]
    messages = [
        {"role": "system", "content": "Je herstelt ongeldige JSON tot precies ÉÉN geldig JSON-object. Alleen JSON."},
        {
            "role": "user",
            "content": (
                f"Parsefout: {err}\n\n"
                "Herschrijf dit naar syntactisch geldige JSON met dezelfde velden inhoudelijk behouden. "
                "Gebruik dubbele quotes. Geen trailing comma.\n\n"
                f"{clipped}"
            ),
        },
    ]
    raw = _chat_extract(client, messages=messages, force_json_format=True)
    return _parse_llm_json(raw)


MULTI_SOURCE_USER_INSTRUCTIONS = """
[Context: de tekst hieronder bevat meerdere achtereenvolgende PDF-bronnen van DEZELFDE patiënt.]
Voeg de inhoud samen tot één JSON volgens het schema:
- patient: één consistent profiel; bij duidelijk tegenstrijdige identiteitsgegevens: kies de meest onderbouwde waarde uit de tekst of zet op null en vermeld kort in extractie_metadata.opmerkingen.
- document (datum, type, ziekenhuis, afdeling): waar mogelijk de belangrijkste of meest recente klinische brief; anders null met toelichting in extractie_metadata.
- medisch: voeg diagnoses, medicatie en allergieën uit alle bronnen samen; geen duplicaten tenzij de bronteksten letterlijk verschillen (dan beide verbatim).
"""


def extract_patient_data(document: dict) -> dict:
    """Extraheer gestructureerde patiëntdata uit document['full_text']."""
    full_text = document.get("full_text", "") or ""

    if not full_text.strip():
        logger.warning("Document bevat geen tekst — extractie overgeslagen")
        return {"extraction_success": False, "extraction_error": "Geen tekst gevonden in document"}

    full_text = _maybe_truncate_document_text(full_text)
    logger.info(f"Extractie starten met {OLLAMA_MODEL}…")

    multi = bool(document.get("multi_source_bundle"))
    if multi:
        logger.info("Multi-bron (meerdere PDF's zelfde patiënt) — extractie op samengevoegde tekst")

    intro = "Extraheer patiëntinformatie uit het medische document."
    if multi:
        intro = (
            "Extraheer patiëntinformatie uit de samengevoegde medische documenten "
            "(meerdere PDF's, zelfde patiënt)."
        )

    user_prompt = f"""{intro}
Antwoord als JSON met exact deze top-level sleutels: patient, document, medisch, extractie_metadata.

Schema (structuur):

{json.dumps(EXTRACTION_SCHEMA, indent=2, ensure_ascii=False)}
{MULTI_SOURCE_USER_INSTRUCTIONS if multi else ""}
Hieronder volgt letterlijke documenttekst (geen deel van het schema):

{full_text}"""

    client = ollama.Client(host=OLLAMA_BASE_URL)
    raw_output = ""

    try:
        raw_output = _chat_extract(
            client,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            force_json_format=True,
        )
        logger.debug(f"Model output ({len(raw_output)} tekens)")

        extracted = _parse_llm_json(raw_output)
        extracted["extraction_success"] = True
        logger.success("Extractie geslaagd")
        return extracted

    except ollama.ResponseError as e:
        logger.error(f"Ollama fout: {e}")
        return {"extraction_success": False, "extraction_error": f"Ollama fout: {str(e)}"}

    except (json.JSONDecodeError, ValueError, TypeError) as e:
        logger.warning(f"JSON mislukt, tweede poging (herstel): {e}")
        broken = raw_output or ""
        try:
            extracted = _repair_json_via_llm(client, broken, str(e))
            extracted["extraction_success"] = True
            logger.success("Extractie gelukt na JSON-herstelronde")
            return extracted
        except Exception as e2:
            logger.error(f"Herstelpoging JSON faalde: {e2}")
            return {
                "extraction_success": False,
                "extraction_error": f"Ongeldige JSON van model: {e}; herstel faalde: {e2}",
            }
