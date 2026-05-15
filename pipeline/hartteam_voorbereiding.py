"""
pipeline/hartteam_voorbereiding.py
-----------------------------------
Genereert een opgemaakte tekst-voorbereiding voor het hartteam op basis van
geëxtraheerde patiëntdata. Geen LLM nodig — puur data-formatting.
"""

from __future__ import annotations

from datetime import datetime


# ── Hulpfuncties ───────────────────────────────────────────────────────────────

def _v(val) -> bool:
    """Heeft dit veld een bruikbare waarde?"""
    if val is None:
        return False
    if isinstance(val, (list, dict)):
        return len(val) > 0
    if isinstance(val, str):
        return val.strip() != ""
    return True  # bool True/False zijn beide een waarde


def _age(geboortedatum: str | None, document_datum: str | None) -> str | None:
    if not geboortedatum or not document_datum:
        return None
    try:
        dob = datetime.strptime(geboortedatum, "%d-%m-%Y").date()
        ref = datetime.strptime(document_datum, "%d-%m-%Y").date()
        age = ref.year - dob.year - ((ref.month, ref.day) < (dob.month, dob.day))
        return str(age)
    except ValueError:
        return None


def _fmt_bool(val: bool | None) -> str | None:
    if val is True:
        return "ja"
    if val is False:
        return "nee"
    return None


def _section(title: str, lines: list[str]) -> str:
    body = "\n".join(l for l in lines if l)
    if not body.strip():
        return ""
    return f"{title}\n{body}"


def _kv(label: str, val, *, indent: int = 2) -> str:
    if not _v(val):
        return ""
    pad = " " * indent
    if isinstance(val, list):
        items = "\n".join(f"{pad}  - {item}" for item in val if item)
        return f"{pad}{label}:\n{items}" if items else ""
    return f"{pad}{label}: {val}"


# ── Secties ────────────────────────────────────────────────────────────────────

def _render_patient_header(data: dict) -> str:
    patient = data.get("patient") or {}
    medisch = data.get("medisch") or {}
    document = data.get("document") or {}

    voornaam = patient.get("voornaam") or ""
    achternaam = patient.get("achternaam") or ""
    naam = " ".join(p for p in [voornaam, achternaam] if p) or None

    leeftijd = _age(patient.get("geboortedatum"), document.get("datum"))
    geslacht = {"M": "man", "V": "vrouw"}.get(patient.get("geslacht") or "", patient.get("geslacht"))
    centrum = medisch.get("verwijzend_ziekenhuis") or document.get("ziekenhuis_van_herkomst")
    opname_type = medisch.get("opname_type")

    parts = [p for p in [naam, f"{leeftijd} jaar" if leeftijd else None,
                         geslacht, centrum, opname_type] if p]
    return ", ".join(parts)


def _render_voorgeschiedenis(medisch: dict) -> str:
    vg = medisch.get("voorgeschiedenis") or {}
    cardiaal = vg.get("cardiaal") or []
    overig = vg.get("overig") or []
    lines = []
    if cardiaal:
        lines.append("  Cardiaal")
        lines.extend(f"    - {item}" for item in cardiaal)
    if overig:
        lines.append("  Overig")
        lines.extend(f"    - {item}" for item in overig)
    return _section("Voorgeschiedenis", lines)


def _render_anamnese(anam: dict) -> str:
    lines = []
    bool_fields = [
        ("pijn_op_de_borst",      "Pijn op de borst"),
        ("pob_bij_inspanning",    "  - bij inspanning"),
        ("pob_in_rust",           "  - in rust"),
        ("dyspnoe",               "Dyspnoe"),
        ("dyspnoe_bij_inspanning","  - bij inspanning"),
        ("palpitaties",           "Palpitaties"),
        ("syncope",               "Syncope/wegraking"),
        ("oedeem",                "Oedeem"),
    ]
    for key, label in bool_fields:
        val = anam.get(key)
        if val is not None:
            lines.append(f"  {label}: {_fmt_bool(val)}")

    for key, label in [("roken", "Roken"), ("alcohol", "Alcohol"),
                       ("drugs", "Drugs"), ("familieanamnese", "Familieanamnese")]:
        val = anam.get(key)
        if _v(val):
            lines.append(f"  {label}: {val}")

    for item in (anam.get("overige") or []):
        lines.append(f"  {item}")

    return _section("Anamnese", lines)


def _render_echo(title: str, echo: dict | None) -> str:
    if not echo:
        return ""

    datum = echo.get("datum")
    header = f"{title}  {datum}" if datum else title
    lines: list[str] = []

    def subsection(name: str, sub: dict | None, fields: list[tuple[str, str]]) -> None:
        if not sub:
            return
        sub_lines = []
        for key, label in fields:
            val = sub.get(key)
            if isinstance(val, list):
                if val:
                    sub_lines.append(f"    {label}:")
                    sub_lines.extend(f"      - {item}" for item in val)
            elif _v(val):
                sub_lines.append(f"    {label}: {val}")
        if sub_lines:
            lines.append(f"  {name}")
            lines.extend(sub_lines)

    subsection("LV", echo.get("LV"), [
        ("dimensies", "Dimensies"), ("LVIDd", "LVIDd"), ("hypertrofie", "Hypertrofie"),
        ("systolische_functie", "Systolische functie"), ("EF", "EF"),
        ("diastolische_functie", "Diastolische functie"), ("E_e_prime", "E/e'"),
        ("E_A_verhouding", "E/A"), ("S_D_verhouding", "S/D"), ("RWBS", "RWBS"),
        ("overige", "Overige"),
    ])
    subsection("RV", echo.get("RV"), [
        ("dimensies", "Dimensies"), ("functie", "Functie"),
        ("TAPSE", "TAPSE"), ("S_prime", "S'"), ("overige", "Overige"),
    ])
    subsection("LA", echo.get("LA"), [
        ("dimensies", "Dimensies"), ("LAVI", "LAVI"),
        ("LAESV_index", "LAESV index"), ("volume", "Volume"), ("overige", "Overige"),
    ])
    subsection("RA", echo.get("RA"), [
        ("dimensies", "Dimensies"), ("RAVI", "RAVI"),
        ("RAESV", "RAESV"), ("area", "Area"), ("overige", "Overige"),
    ])

    kleppen = echo.get("kleppen") or {}
    klep_lines: list[str] = []

    def klep(name: str, fields: list[tuple[str, str]]) -> None:
        k = kleppen.get(name) or {}
        k_lines = []
        for key, label in fields:
            val = k.get(key)
            if isinstance(val, list):
                if val:
                    k_lines.append(f"      {label}:")
                    k_lines.extend(f"        - {item}" for item in val)
            elif _v(val):
                k_lines.append(f"      {label}: {val}")
        if k_lines:
            klep_lines.append(f"    {name}")
            klep_lines.extend(k_lines)

    klep("AOV", [("morfologie", "Morfologie"), ("opening", "Opening"),
                 ("gradienten", "Gradiënten"), ("insufficientie", "Insufficiëntie"),
                 ("AI_P1_2t", "AI P1/2t"), ("AI_end_d_velocity", "AI end-d velocity"),
                 ("overige", "Overige")])
    klep("MV",  [("morfologie", "Morfologie"), ("mean_PG", "Mean PG"),
                 ("insufficientie", "Insufficiëntie"),
                 ("calcificatie_annulus", "Calcificatie annulus"), ("overige", "Overige")])
    klep("TV",  [("insufficientie", "Insufficiëntie"), ("Ti_Vmax", "Ti Vmax"),
                 ("max_PG", "Max PG"), ("overige", "Overige")])
    klep("PV",  [("insufficientie", "Insufficiëntie"), ("overige", "Overige")])

    if klep_lines:
        lines.append("  Kleppen")
        lines.extend(klep_lines)

    subsection("Aorta", echo.get("aorta"), [
        ("AO_dimensies", "Dimensies"), ("aorta_ascendens", "Ascendens"),
        ("AO_root", "Root"), ("AO_boog", "Boog"), ("AO_abdominalis", "Abdominalis"),
    ])
    subsection("VCI", echo.get("VCI"), [
        ("collaps", "Collaps"), ("geschatte_CVD", "Geschatte CVD"),
        ("sPAP", "sPAP"), ("overige", "Overige"),
    ])
    subsection("PHT", echo.get("PHT"), [
        ("sPAP", "sPAP"), ("secundaire_aanwijzingen", "Secundaire aanwijzingen"),
    ])

    for key, label in [("PE", "PE"), ("ritme", "Ritme"), ("hartfrequentie", "Hartfrequentie")]:
        line = _kv(label, echo.get(key))
        if line:
            lines.append(line)

    ov = echo.get("overige_bevindingen") or []
    if ov:
        lines.append("  Overige bevindingen")
        lines.extend(f"    - {item}" for item in ov)

    conclusie = echo.get("conclusie") or []
    if conclusie:
        lines.append("  Conclusie")
        lines.extend(f"    - {item}" for item in conclusie)

    if not lines:
        return ""
    return header + "\n" + "\n".join(lines)


def _render_cag(cag: dict | None) -> str:
    if not cag:
        return ""

    datum = cag.get("datum")
    header = f"CAG  {datum}" if datum else "CAG"
    lines: list[str] = []

    if _v(cag.get("dominantie")):
        lines.append(f"  Dominantie: {cag['dominantie']}")

    nc = cag.get("natieve_coronairen") or {}
    for vat in ["RCA", "LM", "RDA", "RCX"]:
        v = nc.get(vat) or {}
        bevinding = v.get("bevinding")
        stenose_loc = v.get("stenose_locatie") or []
        stenose_gr = v.get("stenose_graad") or []
        overige = v.get("overige") or []
        vat_lines = []
        if _v(bevinding):
            vat_lines.append(f"    Bevinding: {bevinding}")
        for i, loc in enumerate(stenose_loc):
            graad = stenose_gr[i] if i < len(stenose_gr) else ""
            vat_lines.append(f"    Stenose: {(loc + ' ' + graad).strip()}")
        vat_lines.extend(f"    {item}" for item in overige)
        if vat_lines:
            lines.append(f"  {vat}")
            lines.extend(vat_lines)

    for key, label in [("collateralen", "Collateralen"),
                       ("hemostase", "Hemostase"),
                       ("complicaties", "Complicaties")]:
        if _v(cag.get(key)):
            lines.append(f"  {label}: {cag[key]}")

    conclusie = cag.get("conclusie") or []
    if conclusie:
        lines.append("  Conclusie")
        lines.extend(f"    - {item}" for item in conclusie)

    beleid = cag.get("beleid") or []
    if beleid:
        lines.append("  Beleid")
        lines.extend(f"    - {item}" for item in beleid)

    ov = cag.get("overige_bevindingen") or []
    if ov:
        lines.append("  Overige bevindingen")
        lines.extend(f"    - {item}" for item in ov)

    if not lines:
        return ""
    return header + "\n" + "\n".join(lines)


def _render_beeldvorming_item(titel: str, item: dict | None) -> str:
    if not item:
        return ""
    regels = item.get("regels") or []
    if not regels:
        return ""
    datum = item.get("datum")
    header = f"{titel}  {datum}" if datum else titel
    body = "\n".join(f"  {r}" for r in regels)
    return f"{header}\n{body}"


def _render_lab(lab: dict | None) -> str:
    if not lab:
        return ""
    kreatinine = lab.get("kreatinine")
    if not _v(kreatinine):
        return ""
    datum = lab.get("datum")
    header = f"Lab  {datum}" if datum else "Lab"
    return f"{header}\n  Kreatinine: {kreatinine}"


def _render_medicatie(medicatie: list | None) -> str:
    if not medicatie:
        return ""
    lines = []
    for med in medicatie:
        naam = med.get("naam") or ""
        dos = med.get("dosering") or ""
        freq = med.get("frequentie") or ""
        parts = [p for p in [naam, dos, freq] if p]
        if parts:
            lines.append(f"  - {' '.join(parts)}")
    return _section("Medicatie", lines)


# ── Publieke interface ─────────────────────────────────────────────────────────

def generate(extracted: dict) -> str:
    """
    Genereer een opgemaakte tekst-voorbereiding voor het hartteam.
    Alleen velden met inhoud worden getoond.

    Parameters
    ----------
    extracted : dict
        Het geëxtraheerde JSON-object (result["extracted"]).

    Returns
    -------
    str
        Opgemaakte platte tekst, klaar om als .txt op te slaan.
    """
    medisch = extracted.get("medisch") or {}
    anamnese = extracted.get("anamnese") or {}
    diagnostiek = extracted.get("diagnostiek") or {}
    lab = extracted.get("laboratorium") or {}

    sections: list[str] = []

    header = _render_patient_header(extracted)
    if header:
        sections.append(header)

    rvv = medisch.get("reden_van_verwijzing")
    if _v(rvv):
        sections.append(f"Reden van verwijzing\n  {rvv}")

    vg = _render_voorgeschiedenis(medisch)
    if vg:
        sections.append(vg)

    anam = _render_anamnese(anamnese)
    if anam:
        sections.append(anam)

    ao_parts: list[str] = []

    ecg = diagnostiek.get("ecg")
    if _v(ecg):
        ao_parts.append(f"ECG\n  {ecg}")

    tte = _render_echo("TTE", diagnostiek.get("TTE"))
    if tte:
        ao_parts.append(tte)

    tee = _render_echo("TEE", diagnostiek.get("TEE"))
    if tee:
        ao_parts.append(tee)

    cag = _render_cag(diagnostiek.get("cag"))
    if cag:
        ao_parts.append(cag)

    bv = diagnostiek.get("beeldvorming") or {}
    for titel, sleutel in [("MRI hart", "mri_hart"), ("CT coronairen", "ct_coronairen"),
                           ("CT thorax", "ct_thorax"), ("PET-CT", "pet_ct")]:
        rendered = _render_beeldvorming_item(titel, bv.get(sleutel))
        if rendered:
            ao_parts.append(rendered)

    for item in (bv.get("overige_beeldvorming") or []):
        rendered = _render_beeldvorming_item(item.get("modaliteit") or "Beeldvorming", item)
        if rendered:
            ao_parts.append(rendered)

    lab_str = _render_lab(lab)
    if lab_str:
        ao_parts.append(lab_str)

    if ao_parts:
        sections.append("Aanvullend onderzoek\n\n" + "\n\n".join(ao_parts))

    med_str = _render_medicatie(medisch.get("medicatie"))
    if med_str:
        sections.append(med_str)

    return "\n\n".join(sections) + "\n"
