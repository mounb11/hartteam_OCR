"""
main.py
-------
Hoofdbestand — koppelt alle stappen van de pipeline aan elkaar.

Gebruik:
  python main.py                    # alle PDFs in input/ inclusief submappen
  python main.py pad/naar/file.pdf  # verwerk één specifieke PDF
  python main.py --test             # gebruik de testdata in tests/test_pdfs/

Pipeline stappen (per PDF):
  1. PDF inlezen       → pdf_processor  (embedded tekst of OCR via Tesseract)
  2. Data extraheren   → extractor      (qwen2.5:14b via Ollama, lokaal)
  3. Valideren         → validator      (BSN 11-proef, verplichte aanbevolen velden)
  4. FHIR mappen       → fhir_mapper    (HL7 FHIR R4 Bundle; JSON-output, geen automatische export)

Optie: meerdere PDF's van dezelfde patiënt samenvoegen (`process_pdfs_merged_same_patient`) — één gezamenlijke tekstbron en één extractie.

Volledige platte tekst staat onder `document.full_text` en `document.pages_verbatim` in het JSON-resultaat,
en optioneel onder `output/*_verbatim.txt` (SAVE_VERBATIM_TEXT_FILE).
"""

from __future__ import annotations

import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Sequence

from loguru import logger

from config import (
    INPUT_DIR,
    OUTPUT_DIR,
    LOG_DIR,
    TEST_PDF_DIR,
    STOP_ON_ERROR,
    PAGE_JOIN_SEPARATOR,
    SAVE_VERBATIM_TEXT_FILE,
)
from pipeline.pdf_processor import extract_text_from_pdf
from pipeline.extractor import extract_patient_data
from pipeline.validator import validate_extracted_data
from pipeline.fhir_mapper import map_to_fhir


def pdf_files_under(root: Path) -> list[Path]:
    """Alle PDFs onder root (recursief), case-insensitief voor de extensie."""
    root = root.resolve()
    if not root.is_dir():
        return []
    paths: list[Path] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() == ".pdf":
            paths.append(p)
    return sorted(paths)


def json_output_slug_for_pdf(pdf_path: Path) -> str:
    """Bestandsnaam voor output JSON: submap__bestand bij PDFs onder input/."""
    try:
        rel = pdf_path.resolve().relative_to(INPUT_DIR.resolve())
        rel_stem = rel.with_suffix("")
        return str(rel_stem).replace("\\", "__").replace("/", "__")
    except ValueError:
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in pdf_path.stem)
        return safe or pdf_path.name


def resolve_output_slug(pdf_path: Path, output_label: str | None = None) -> str:
    """
    Basisnaam voor output-JSON/verbatim. Bij UI-uploads: `output_label` = originele bestandsnaam,
    zodat de output niet naar een tijdelijke random naam wijst.
    """
    if output_label:
        stem = Path(output_label).stem
        clean = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in stem)
        return clean or "document"
    return json_output_slug_for_pdf(pdf_path)


def save_verbatim_text_sidecar(
    pdf_path: Path, full_text: str, *, output_slug: str | None = None
) -> Path | None:
    """Schrijf identieke platte tekst als .txt naast JSON (uit te zetten via SAVE_VERBATIM_TEXT_FILE=0)."""
    if not SAVE_VERBATIM_TEXT_FILE:
        return None
    slug = output_slug if output_slug is not None else json_output_slug_for_pdf(pdf_path)
    path = OUTPUT_DIR / f"{slug}_verbatim.txt"
    path.write_text(full_text, encoding="utf-8")
    logger.info(f"Volledige tekstextractie (verbatim): {path.name}")
    return path


def setup_logging():
    """Stel logging in: naar console én naar een logbestand."""
    logger.remove()  # Verwijder standaard handler

    # Console: leesbaar formaat
    logger.add(
        sys.stdout,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        level="INFO",
        colorize=True,
    )

    # Logbestand: volledig formaat met tijdstempel
    log_file = LOG_DIR / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger.add(
        str(log_file),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} | {message}",
        level="DEBUG",
        rotation="10 MB",  # Nieuw bestand na 10 MB
    )

    logger.info(f"Log opgeslagen in: {log_file}")


def build_merged_patient_document(
    labelled_parts: list[tuple[str, dict]],
) -> dict:
    """
    Combineer tekstextracties van meerdere PDF's tot één `document`-dict voor het LLM.

    Elk element is `(label_zoals_bestandsnaam, dict van extract_text_from_pdf)`.
    """
    if len(labelled_parts) < 2:
        raise ValueError("Minimaal twee PDF-bronnen nodig om samen te voegen.")

    total_sources = len(labelled_parts)
    prelude = (
        "[Context voor interpretatie: de blokken hieronder zijn opeenvolgende PDF-bronnen "
        "van dezelfde patiënt.]\n\n"
    )

    blocks: list[str] = []
    pages_verbatim: list[dict] = []
    methods_flat: list[str] = []
    source_meta: list[dict] = []
    bundle_page = 0

    between = PAGE_JOIN_SEPARATOR

    for i, (label, doc) in enumerate(labelled_parts, start=1):
        header = f"===== BRON {i}/{total_sources}: {label} ({len(doc['pages'])} pag.) ====="
        blocks.append(header + "\n" + doc["full_text"])
        source_meta.append(
            {
                "label": label,
                "pdf_path": doc.get("path"),
                "pages": len(doc["pages"]),
            }
        )

        for p in doc["pages"]:
            bundle_page += 1
            pages_verbatim.append(
                {
                    "bundle_page_number": bundle_page,
                    "source_index": i,
                    "source_label": label,
                    "page_number_in_source": p["page_number"],
                    "method": p["method"],
                    "text": p["text"],
                }
            )
            methods_flat.append(p["method"])

    full_text = prelude + between.join(blocks)

    return {
        "merged_same_patient": True,
        "multi_source_bundle": True,
        "merge_source_labels": [lab for lab, _ in labelled_parts],
        "sources": source_meta,
        "path": f"merged:{total_sources}_pdfs",
        "pages": bundle_page,
        "page_separator": PAGE_JOIN_SEPARATOR,
        "methods": methods_flat,
        "total_chars": len(full_text),
        "full_text": full_text,
        "pages_verbatim": pages_verbatim,
    }


def resolve_merged_output_slug(
    source_labels: Sequence[str],
    output_label: str | None,
) -> str:
    """Basisnaam voor JSON output bij samengevoegd patiëntdossier."""
    if output_label and output_label.strip():
        lbl = output_label.strip()
        if not lbl.lower().endswith(".pdf"):
            lbl = f"{lbl}.pdf"
        return resolve_output_slug(Path("."), lbl)

    stems: list[str] = []
    for lab in source_labels[:6]:
        piece = resolve_output_slug(
            Path("."),
            lab if lab.lower().endswith(".pdf") else f"{lab}.pdf",
        )
        stems.append(piece)
    joined = "__".join(stems)
    if len(source_labels) > 6:
        joined += f"__plus{len(source_labels) - 6}more"

    return joined[:180] if len(joined) > 180 else joined


def _pipeline_llm_validate_fhir(
    result: dict,
    *,
    pdf_path: Path,
    out_slug: str,
    document_for_llm: dict,
    completion_log_hint: str,
) -> dict:
    """Delen 2–4: LLM-extractie, validatie, FHIR (gemeenschappelijk voor enkel/samengevoegd)."""

    logger.info("Stap 2/4: Data extraheren met Ollama")
    try:
        extracted = extract_patient_data(document_for_llm)
        result["extracted"] = extracted
    except Exception as e:
        logger.error(f"Stap 2 mislukt: {e}")
        result["status"] = "failed"
        result["errors"].append(f"LLM extractie: {str(e)}")
        return result

    logger.info("Stap 3/4: Valideren")
    validation = validate_extracted_data(extracted)
    result["validation"] = {
        "is_valid": validation.is_valid,
        "needs_human_review": validation.needs_human_review,
        "errors": validation.errors,
        "warnings": validation.warnings,
    }

    if not validation.is_valid:
        logger.error(f"Validatie gefaald: {validation.errors}")
        result["status"] = "validation_failed"
        _save_output(result, pdf_path, suffix="review_required", output_slug=out_slug)
        return result

    if validation.needs_human_review:
        logger.warning("Document gemarkeerd voor menselijke review")
        result["status"] = "needs_review"
        _save_output(result, pdf_path, suffix="needs_review", output_slug=out_slug)
        return result

    logger.info("Stap 4/4: Mappen naar FHIR R4")
    try:
        fhir_bundle = map_to_fhir(extracted)
        result["fhir_bundle"] = fhir_bundle
    except Exception as e:
        logger.error(f"Stap 4 mislukt: {e}")
        result["status"] = "failed"
        result["errors"].append(f"FHIR mapping: {str(e)}")
        return result

    result["status"] = "completed"
    logger.success(f"Volledig verwerkt — {completion_log_hint}")
    _save_output(result, pdf_path, output_slug=out_slug)
    return result


def process_pdfs_merged_same_patient(
    pdf_paths: Sequence[Path],
    *,
    source_labels: Sequence[str] | None = None,
    output_label: str | None = None,
) -> dict:
    """
    Twee of meer PDF's van dezelfde patiënt: tekst (in opgegeven volgorde) samenvoegen,
    één keer extractie/validatie/FHIR.

    `source_labels`: o.a. originele uploadnamen; standaard `pdf_paths[i].name`.
    `output_label`: optionele basisnaam voor output-JSON (zonder pad).
    """
    paths = list(pdf_paths)
    if len(paths) < 2:
        raise ValueError("Minimaal twee PDF-paden nodig voor samenvoegen.")

    labels = list(source_labels) if source_labels is not None else [p.name for p in paths]
    if len(labels) != len(paths):
        raise ValueError("source_labels en pdf_paths moeten dezelfde lengte hebben.")

    out_slug = resolve_merged_output_slug(labels, output_label)

    logger.info(f"{'─'*60}")
    logger.info(f"Samengevoegd dossier: {len(paths)} PDF(s) → één extractie (output: {out_slug})")
    logger.info(f"{'─'*60}")

    result: dict = {
        "pdf": "merged_same_patient_bundle",
        "pdfs": [str(p) for p in paths],
        "merged_same_patient": True,
        "merge_source_labels": labels,
        "timestamp": datetime.now().isoformat(),
        "status": "pending",
        "document": None,
        "extracted": None,
        "validation": None,
        "fhir_bundle": None,
        "errors": [],
    }

    labelled_docs: list[tuple[str, dict]] = []
    try:
        for path, lab in zip(paths, labels):
            logger.info(f"Stap 1a: PDF → tekst — {lab}")
            labelled_docs.append((lab, extract_text_from_pdf(path)))
    except Exception as e:
        logger.error(f"PDF-extractie in merge-flow mislukt: {e}")
        result["status"] = "failed"
        result["errors"].append(f"PDF extractie (merge): {str(e)}")
        return result

    merged_doc = build_merged_patient_document(labelled_docs)
    save_verbatim_text_sidecar(
        paths[0],
        merged_doc["full_text"],
        output_slug=out_slug,
    )

    result["document"] = merged_doc
    pdf_path_for_save = paths[0]

    return _pipeline_llm_validate_fhir(
        result,
        pdf_path=pdf_path_for_save,
        out_slug=out_slug,
        document_for_llm=merged_doc,
        completion_log_hint=f"samengevoegd dossier ({len(paths)} PDF's)",
    )


# ── PDF verwerken ──────────────────────────────────────────────────────────────

def process_pdf(pdf_path: Path, *, output_label: str | None = None) -> dict:
    """
    Verwerk één PDF door de volledige pipeline (tekst → LLM → validatie → FHIR-JSON lokaal).

    `output_label`: optioneel originele bestandsnaam (bijv. Streamlit-upload) voor nette output-slugs.

    Returns een resultaat-dict met status en alle tussenresultaten,
    zodat je achteraf kunt zien wat er is gebeurd.
    """
    out_slug = resolve_output_slug(pdf_path, output_label)

    logger.info(f"{'─'*60}")
    logger.info(f"PDF: {pdf_path.name}")
    logger.info(f"{'─'*60}")

    result = {
        "pdf": str(pdf_path),
        "timestamp": datetime.now().isoformat(),
        "status": "pending",
        "document": None,
        "extracted": None,
        "validation": None,
        "fhir_bundle": None,
        "errors": [],
    }

    # ── Stap 1: PDF → tekst ────────────────────────────────────────────────────
    logger.info("Stap 1/4: PDF inlezen en tekst extraheren")
    try:
        document = extract_text_from_pdf(pdf_path)
        full_plain = document["full_text"]
        save_verbatim_text_sidecar(pdf_path, full_plain, output_slug=out_slug)

        result["document"] = {
            "pages": len(document["pages"]),
            "page_separator": PAGE_JOIN_SEPARATOR,
            "methods": [p["method"] for p in document["pages"]],
            "total_chars": len(full_plain),
            "full_text": full_plain,
            "pages_verbatim": [
                {"page_number": p["page_number"], "method": p["method"], "text": p["text"]}
                for p in document["pages"]
            ],
        }
    except Exception as e:
        logger.error(f"Stap 1 mislukt: {e}")
        result["status"] = "failed"
        result["errors"].append(f"PDF extractie: {str(e)}")
        return result

    document_for_llm = document

    return _pipeline_llm_validate_fhir(
        result,
        pdf_path=pdf_path,
        out_slug=out_slug,
        document_for_llm=document_for_llm,
        completion_log_hint=pdf_path.name,
    )


def _save_output(
    result: dict,
    pdf_path: Path,
    suffix: str = "output",
    *,
    output_slug: str | None = None,
):
    """Sla het resultaat op als JSON in de output/ map."""
    slug = output_slug if output_slug is not None else json_output_slug_for_pdf(pdf_path)
    output_file = OUTPUT_DIR / f"{slug}_{suffix}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info(f"Resultaat opgeslagen: {output_file.name}")


def validation_result_to_dict(v) -> dict:
    return {
        "is_valid": v.is_valid,
        "needs_human_review": v.needs_human_review,
        "errors": v.errors,
        "warnings": v.warnings,
    }


def merge_extracted_and_finalize(
    result: dict,
    pdf_path_for_slug: Path,
    extracted: dict,
    *,
    accept_warnings_for_fhir: bool = False,
    save: bool = True,
    output_slug: str | None = None,
) -> dict:
    """
    Vervangt `extracted` in een bestaande pipeline-resultaat, voert validatie/FHIR opnieuw uit.

    - Bij validatiefouten (`is_valid` False): geen FHIR.
    - Bij waarschuwingen: FHIR alleen als `accept_warnings_for_fhir` True (nadat je velden handmatig hebt gecorrigeerd).

    Werkt ook als `extracted` nog geen extraction_success heeft; die wordt hier op True gezet.
    """
    out = dict(result)
    extracted = dict(extracted)
    extracted["extraction_success"] = extracted.get("extraction_success", True)

    validation = validate_extracted_data(extracted)
    vd = validation_result_to_dict(validation)

    out["extracted"] = extracted
    out["validation"] = vd
    out["fhir_bundle"] = None
    out.setdefault("errors", [])

    if not validation.is_valid:
        logger.error(f"Na handmatige wijziging nog steeds fouten: {validation.errors}")
        out["status"] = "validation_failed"
        if save:
            _save_output(
                out, pdf_path_for_slug, suffix="review_required", output_slug=output_slug
            )
        return out

    allow_fhir = (not validation.needs_human_review) or accept_warnings_for_fhir

    if not allow_fhir:
        out["status"] = "needs_review"
        if save:
            _save_output(out, pdf_path_for_slug, suffix="needs_review", output_slug=output_slug)
        return out

    try:
        out["fhir_bundle"] = map_to_fhir(extracted)
        out["status"] = "completed"
        logger.success(f"Na handmatige wijziging: FHIR klaar ({pdf_path_for_slug.name})")
        if save:
            _save_output(out, pdf_path_for_slug, output_slug=output_slug)
    except Exception as e:
        logger.error(f"FHIR na handmatige wijziging mislukt: {e}")
        out["status"] = "failed"
        err = f"FHIR mapping: {str(e)}"
        out["errors"] = list(out.get("errors") or []) + [err]

    return out
# ── Pipeline voor een map PDFs ─────────────────────────────────────────────────

def run_pipeline(pdf_dir: Path):
    """Verwerk alle PDFs onder pdf_dir (inclusief alle submappen)."""
    pdf_files = pdf_files_under(pdf_dir)

    if not pdf_files:
        logger.warning(f"Geen PDFs gevonden onder {pdf_dir} (recursief)")
        return

    logger.info(f"{len(pdf_files)} PDF(s) gevonden onder {pdf_dir}")

    results = {"total": len(pdf_files), "completed": 0, "failed": 0, "needs_review": 0}

    for pdf_path in pdf_files:
        try:
            result = process_pdf(pdf_path)

            if result["status"] == "completed":
                results["completed"] += 1
            elif result["status"] == "needs_review":
                results["needs_review"] += 1
            else:
                results["failed"] += 1

        except Exception as e:
            logger.error(f"Onverwachte fout bij {pdf_path.name}: {e}")
            results["failed"] += 1
            if STOP_ON_ERROR:
                raise

    logger.info(f"{'='*60}")
    logger.info(f"Pipeline klaar:")
    logger.info(f"  Verwerkt:       {results['completed']}/{results['total']}")
    logger.info(f"  Review nodig:   {results['needs_review']}/{results['total']}")
    logger.info(f"  Mislukt:        {results['failed']}/{results['total']}")
    logger.info(f"  Output in:      {OUTPUT_DIR}")
    logger.info(f"{'='*60}")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    setup_logging()

    args = sys.argv[1:]

    if "--test" in args:
        logger.info("Testmodus")
        run_pipeline(TEST_PDF_DIR)

    elif args and not args[0].startswith("--"):
        # Enkel bestand opgegeven
        pdf_path = Path(args[0])
        if not pdf_path.exists():
            logger.error(f"Bestand niet gevonden: {pdf_path}")
            sys.exit(1)
        process_pdf(pdf_path)

    else:
        run_pipeline(INPUT_DIR)
