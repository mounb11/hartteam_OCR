"""
Lokale web-UI voor de Hartteam OCR-pipeline (Streamlit).

Starten:
  streamlit run ui_app.py
of dubbelklik op Start-Hartteam-OCR.bat / de snelkoppeling op het bureaublad.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import OUTPUT_DIR  # noqa: E402
from main import (  # noqa: E402
    merge_extracted_and_finalize,
    process_pdf,
    process_pdfs_merged_same_patient,
    resolve_merged_output_slug,
    resolve_output_slug,
    setup_logging,
)


def _init_logging_once() -> None:
    if st.session_state.get("_log_done"):
        return
    setup_logging()
    st.session_state["_log_done"] = True


def _norm_optional_str(text: str) -> str | None:
    stripped = text.strip()
    return stripped if stripped else None


def _parse_json_array_field(raw: str, field_label: str) -> list | None:
    raw = raw.strip()
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        st.error(f"Ongeldige JSON bij {field_label}: {e}")
        return None
    if not isinstance(data, list):
        st.error(f"{field_label}: verwacht een JSON-array (lijst), geen {type(data).__name__}.")
        return None
    return data


def _build_extracted_from_form(
    base: dict,
    *,
    p_achternaam: str,
    p_voornaam: str,
    p_gebdat: str,
    p_bsn: str,
    p_geslacht: str,
    p_adres: str,
    p_pc: str,
    p_woon: str,
    d_datum: str,
    d_type: str,
    d_ziek: str,
    d_afd: str,
    m_hoofd: str,
    m_barts: str,
    m_verw_arts: str,
    m_reden: str,
    m_neven_json: str,
    m_med_json: str,
    m_all_json: str,
    em_miss: str,
    em_opm: str,
) -> dict | None:
    out = copy.deepcopy(base) if isinstance(base, dict) else {}

    nev = _parse_json_array_field(m_neven_json, "Nevendiagnoses")
    meds = _parse_json_array_field(m_med_json, "Medicatie")
    allerg = _parse_json_array_field(m_all_json, "Allergieën")
    if nev is None or meds is None or allerg is None:
        return None

    miss_raw = em_miss.strip()
    missing_fields: list[str]
    if not miss_raw:
        missing_fields = []
    else:
        try:
            mf = json.loads(miss_raw)
        except json.JSONDecodeError:
            missing_fields = [s.strip() for s in miss_raw.split(",") if s.strip()]
        else:
            if not isinstance(mf, list):
                st.error("Ontbrekende velden: geef een JSON-array of kommagescheiden tekst.")
                return None
            missing_fields = [str(x) for x in mf]

    out["patient"] = {
        "achternaam": _norm_optional_str(p_achternaam),
        "voornaam": _norm_optional_str(p_voornaam),
        "geboortedatum": _norm_optional_str(p_gebdat),
        "bsn": _norm_optional_str(p_bsn),
        "geslacht": _norm_optional_str(p_geslacht),
        "adres": _norm_optional_str(p_adres),
        "postcode": _norm_optional_str(p_pc),
        "woonplaats": _norm_optional_str(p_woon),
    }
    out["document"] = {
        "datum": _norm_optional_str(d_datum),
        "type": _norm_optional_str(d_type),
        "ziekenhuis_van_herkomst": _norm_optional_str(d_ziek),
        "afdeling": _norm_optional_str(d_afd),
    }
    out["medisch"] = {
        "hoofddiagnose": _norm_optional_str(m_hoofd),
        "nevendiagnoses": nev,
        "medicatie": meds,
        "allergieën": allerg,
        "behandelend_arts": _norm_optional_str(m_barts),
        "verwijzend_arts": _norm_optional_str(m_verw_arts),
        "reden_van_verwijzing": _norm_optional_str(m_reden),
    }
    out["extractie_metadata"] = {
        "ontbrekende_velden": missing_fields,
        "opmerkingen": _norm_optional_str(em_opm),
    }
    out["extraction_success"] = True
    return out


def _run_pdf_bytes(name: str, data: bytes) -> dict:
    with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)
    try:
        return process_pdf(tmp_path, output_label=name)
    finally:
        tmp_path.unlink(missing_ok=True)


def _run_merged_pdf_uploads(
    files: list,
    dossier_naam: str,
) -> tuple[dict, str, list[str]]:
    """
    Schrijf uploads naar tijdelijke paden, voer merge-pipeline uit, ruim op.

    Returns: (result, output_slug, source_names)
    """
    names = [f.name for f in files]
    out_slug = resolve_merged_output_slug(names, dossier_naam or None)
    tmp_paths: list[Path] = []
    try:
        for f in files:
            with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(f.getbuffer().tobytes())
                tmp_paths.append(Path(tmp.name))
        result = process_pdfs_merged_same_patient(
            tmp_paths,
            source_labels=names,
            output_label=dossier_naam or None,
        )
    finally:
        for p in tmp_paths:
            p.unlink(missing_ok=True)
    return result, out_slug, names


def _render_result_summary(result: dict) -> None:
    status = result.get("status", "")
    if status == "completed":
        st.success(f"Status: **{status}**")
    elif status == "needs_review":
        st.warning("Status: **needs_review** — waarschuwingen, controle aanbevolen.")
    elif status == "validation_failed":
        st.error("Status: **validation_failed** — harde validatiefouten.")
    else:
        st.error(f"Status: **{status}**")

    if result.get("validation"):
        v = result["validation"]
        st.caption(f"Validatie geldig: **{'ja' if v.get('is_valid') else 'nee'}**")
        if v.get("errors"):
            st.error("Fouten: " + "; ".join(v["errors"]))
        if v.get("warnings"):
            st.warning("Waarschuwingen: " + "; ".join(v["warnings"]))


def _render_json_tabs(result: dict, download_name: str) -> None:
    tabs = st.tabs(["Volledige JSON", "Geëxtraheerde data", "FHIR-bundle"])
    with tabs[0]:
        st.download_button(
            "Download JSON",
            data=json.dumps(result, indent=2, ensure_ascii=False),
            file_name=download_name,
            mime="application/json",
        )
        st.json(result)
    with tabs[1]:
        st.json(result.get("extracted") or {})
    with tabs[2]:
        st.json(result.get("fhir_bundle") or {})


st.set_page_config(
    page_title="Hartteam OCR",
    page_icon="📄",
    layout="wide",
)

_init_logging_once()

st.title("Hartteam OCR")
st.caption(
    "PDF → tekst/OCR → Ollama-extractie → validatie → FHIR (lokaal JSON). "
    "Zorg dat Ollama draait en het model geïnstalleerd is (`ollama pull qwen2.5:14b`). "
    "**Tip:** kopiëren/plakken uit een PDF geeft vaak kale rommel door kapotte fonts; deze app gebruikt embedded tekst of OCR — niet je klembord."
)

tab_run, tab_out = st.tabs(["Verwerken", "Output doorzoeken"])

with tab_run:
    st.markdown(
        "**Of via CLI:** zet PDF’s in `input/` en run `python main.py` "
        "(submappen worden meegenomen)."
    )

    mode = st.radio(
        "Modus",
        [
            "Enkel bestand",
            "Bulk (PDF’s los van elkaar)",
            "Eén patiënt (PDF’s samenvoegen)",
        ],
        horizontal=True,
    )

    if mode == "Enkel bestand":
        uf = st.file_uploader("PDF uploaden", type=["pdf"])
        single_go = st.button("Verwerk PDF", type="primary", disabled=uf is None)

        if single_go and uf is not None:
            with st.spinner("Pipeline bezig (LLM kan ~30–90 s duren)…"):
                result = _run_pdf_bytes(uf.name, uf.getbuffer().tobytes())
            slug = resolve_output_slug(Path("."), uf.name)
            batch = [{"name": uf.name, "slug": slug, "result": result}]
            st.session_state["last_batch"] = batch
            _render_result_summary(result)
            _render_json_tabs(result, "hartteam_result.json")

    elif mode == "Bulk (PDF’s los van elkaar)":
        files = st.file_uploader(
            "PDF’s uploaden (meerdere toegestaan)",
            type=["pdf"],
            accept_multiple_files=True,
        )
        bulk_go = st.button(
            "Verwerk alle PDF’s",
            type="primary",
            disabled=not files,
        )

        if bulk_go and files:
            results: list[dict] = []
            bar = st.progress(0.0, text="Start…")
            for i, uf in enumerate(files):
                bar.progress(
                    i / len(files),
                    text=f"{i + 1}/{len(files)} — {uf.name}",
                )
                with st.spinner(f"Bezig met {uf.name}…"):
                    r = _run_pdf_bytes(uf.name, uf.getbuffer().tobytes())
                results.append(
                    {
                        "name": uf.name,
                        "slug": resolve_output_slug(Path("."), uf.name),
                        "result": r,
                    }
                )
            bar.progress(1.0, text="Klaar")
            st.session_state["last_batch"] = results

            st.subheader("Overzicht")
            for row in results:
                st.markdown(
                    f"- **{row['name']}** — `{row['result'].get('status', '?')}`"
                )

            for row in results:
                with st.expander(f"Inhoud — {row['name']}", expanded=False):
                    _render_result_summary(row["result"])
                    safe = "".join(c if c.isalnum() or c in ".-_" else "_" for c in row["name"])
                    _render_json_tabs(row["result"], f"{safe}_result.json")

    else:
        st.caption(
            "Upload **minstens twee** PDF’s **in de gewenste volgorde** (bijv. oud → nieuw). "
            "Ze worden tot één tekstbron geplakt; het model levert **één** JSON en **één** FHIR-bundle."
        )
        merge_files = st.file_uploader(
            "PDF’s van dezelfde patiënt",
            type=["pdf"],
            accept_multiple_files=True,
            key="merge_same_patient",
        )
        dossier = st.text_input(
            "Optioneel: naam voor outputbestand (zonder extensie)",
            placeholder="bv. Jansen_2026_dossier",
            key="merge_dossier_name",
        )
        n_merge = len(merge_files) if merge_files else 0
        merge_go = st.button(
            "Samenvoegen & verwerk",
            type="primary",
            disabled=n_merge < 2,
        )

        if merge_go and merge_files and n_merge >= 2:
            dossier_naam = (dossier or "").strip()
            with st.spinner(
                "PDF’s naar tekst, samenvoegen, daarna extractie — LLM kan lang duren…"
            ):
                result, slug, sources = _run_merged_pdf_uploads(
                    list(merge_files),
                    dossier_naam,
                )
            label = dossier_naam or f"Samengevoegd ({n_merge} PDF’s)"
            st.session_state["last_batch"] = [
                {
                    "name": label,
                    "slug": slug,
                    "result": result,
                    "is_merged_bundle": True,
                    "merge_sources": sources,
                }
            ]
            _render_result_summary(result)
            _render_json_tabs(result, "hartteam_merged_bundle.json")

    # ── Handmatige correctie (batch uit laatste run) ──────────────────────────
    batch = st.session_state.get("last_batch") or []
    if batch:
        st.divider()
        st.subheader("Handmatig velden aanpassen")

        ix = st.selectbox(
            "Document",
            range(len(batch)),
            format_func=lambda i: (
                f'{batch[i]["name"]}  (bronnen: {", ".join(batch[i].get("merge_sources") or [])})'
                if batch[i].get("is_merged_bundle")
                else batch[i]["name"]
            ),
        )

        sel = batch[ix]
        res = sel["result"]
        ext_raw = res.get("extracted")
        slug = sel["slug"]

        if not isinstance(ext_raw, dict) or ext_raw.get("extraction_success") is not True:
            st.info(
                "Geen succesvol geëxtraheerde blob om te bewerken — alleen beschikbaar na geslaagde LLM-json."
            )
        else:
            p = ext_raw.get("patient") or {}
            d = ext_raw.get("document") or {}
            m = ext_raw.get("medisch") or {}
            em = ext_raw.get("extractie_metadata") or {}

            with st.form(f"manual_fields_{ix}_{slug}"):
                st.markdown("**Patiënt**")
                c1, c2 = st.columns(2)
                with c1:
                    p_an = st.text_input("Achternaam", value=p.get("achternaam") or "")
                    p_vn = st.text_input("Voornaam", value=p.get("voornaam") or "")
                    p_gb = st.text_input("Geboortedatum (DD-MM-YYYY)", value=p.get("geboortedatum") or "")
                    p_bs = st.text_input("BSN", value=p.get("bsn") or "")
                with c2:
                    p_gs = st.text_input("Geslacht (M/V)", value=p.get("geslacht") or "")
                    p_ad = st.text_input("Adres", value=p.get("adres") or "")
                    p_pc = st.text_input("Postcode", value=p.get("postcode") or "")
                    p_wp = st.text_input("Woonplaats", value=p.get("woonplaats") or "")

                st.markdown("**Document**")
                c3, c4 = st.columns(2)
                with c3:
                    d_dm = st.text_input("Datum (DD-MM-YYYY)", value=d.get("datum") or "")
                    d_tp = st.text_input("Type", value=d.get("type") or "")
                with c4:
                    d_zk = st.text_input(
                        "Ziekenhuis van herkomst",
                        value=d.get("ziekenhuis_van_herkomst") or "",
                    )
                    d_af = st.text_input("Afdeling", value=d.get("afdeling") or "")

                st.markdown("**Medisch**")
                m_hoofd = st.text_input(
                    "Hoofddiagnose", value=m.get("hoofddiagnose") or ""
                )
                m_be = st.text_input(
                    "Behandelend arts", value=m.get("behandelend_arts") or ""
                )
                m_va = st.text_input(
                    "Verwijzend arts", value=m.get("verwijzend_arts") or ""
                )
                m_rd = st.text_input(
                    "Reden verwijzing", value=m.get("reden_van_verwijzing") or ""
                )
                json_defaults = dict(indent=2, ensure_ascii=False)
                j_neven = json.dumps(m.get("nevendiagnoses") or [], **json_defaults)
                j_med = json.dumps(m.get("medicatie") or [], **json_defaults)
                j_all = json.dumps(m.get("allergieën") or [], **json_defaults)
                m_neven = st.text_area("Nevendiagnoses (JSON-array)", value=j_neven)
                m_medica = st.text_area("Medicatie (JSON-array van objecten)", value=j_med)
                m_allrg = st.text_area("Allergieën (JSON-array)", value=j_all)

                st.markdown("**Extractiemetadata**")
                em_miss = st.text_area(
                    "Ontbrekende velden — JSON-array of kommagescheiden tekst",
                    value=json.dumps(em.get("ontbrekende_velden") or [], **json_defaults),
                )
                em_opm = st.text_input(
                    "Opmerkingen", value=em.get("opmerkingen") or ""
                )

                accept_w = st.checkbox(
                    "FHIR toch genereren ondanks waarschuwingen (niet bij harde fouten)",
                    value=False,
                )

                submitted = st.form_submit_button(
                    "Toepassen → opnieuw valideren / FHIR", type="primary"
                )

            if submitted:
                new_ext = _build_extracted_from_form(
                    ext_raw,
                    p_achternaam=p_an,
                    p_voornaam=p_vn,
                    p_gebdat=p_gb,
                    p_bsn=p_bs,
                    p_geslacht=p_gs,
                    p_adres=p_ad,
                    p_pc=p_pc,
                    p_woon=p_wp,
                    d_datum=d_dm,
                    d_type=d_tp,
                    d_ziek=d_zk,
                    d_afd=d_af,
                    m_hoofd=m_hoofd,
                    m_barts=m_be,
                    m_verw_arts=m_va,
                    m_reden=m_rd,
                    m_neven_json=m_neven,
                    m_med_json=m_medica,
                    m_all_json=m_allrg,
                    em_miss=em_miss,
                    em_opm=em_opm,
                )
                if new_ext is not None:
                    path_for_finalize = Path(
                        (sel.get("merge_sources") or [sel["name"]])[0]
                    )
                    updated = merge_extracted_and_finalize(
                        res,
                        path_for_finalize,
                        new_ext,
                        accept_warnings_for_fhir=accept_w,
                        output_slug=slug,
                    )
                    st.session_state["last_batch"][ix]["result"] = updated
                    st.session_state["last_batch"][ix]["slug"] = slug
                    _render_result_summary(updated)
                    st.rerun()


with tab_out:
    st.subheader("JSON in output/")
    st.caption(str(OUTPUT_DIR.resolve()))

    needle = (
        st.text_input(
            "Filter (substring in bestandsnaam of in JSON-inhoud, leeg = alles)",
            "",
        )
        .strip()
        .lower()
    )

    all_json = sorted(OUTPUT_DIR.glob("*.json"), key=lambda p: p.name.lower())
    filtered: list[Path] = []
    for jp in all_json:
        name_l = jp.name.lower()
        if not needle:
            filtered.append(jp)
            continue
        if needle in name_l:
            filtered.append(jp)
            continue
        try:
            body = jp.read_text(encoding="utf-8").lower()
        except OSError:
            continue
        if needle in body:
            filtered.append(jp)

    if not filtered:
        st.warning("Geen bestanden gevonden (of niets onder filter).")
    else:
        pick = st.selectbox(
            "Bestand",
            filtered,
            format_func=lambda p: p.name,
        )
        try:
            raw = pick.read_text(encoding="utf-8")
            blob = json.loads(raw)
        except (OSError, json.JSONDecodeError) as e:
            st.error(f"Kan JSON niet lezen: {e}")
        else:
            st.download_button(
                "Download dit bestand",
                data=raw,
                file_name=pick.name,
                mime="application/json",
            )
            st.json(blob)

log_hint = PROJECT_ROOT / "logs"
st.caption(f"Logbestanden: `{log_hint}`")
