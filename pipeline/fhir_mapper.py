"""
pipeline/fhir_mapper.py
-----------------------
Stap 4: Geëxtraheerde data omzetten naar HL7 FHIR R4 resources.

Wat is FHIR?
  HL7 FHIR (Fast Healthcare Interoperability Resources) is de internationale standaard
  voor het uitwisselen van medische data. HiX (ChipSoft) ondersteunt FHIR R4.

Wat we hier doen:
  De geëxtraheerde JSON (eigen formaat) → FHIR Resources (standaardformaat):
    - Patient          → demografische gegevens van de patiënt
    - Condition        → diagnoses
    - MedicationStatement → medicatie

We bouwen de FHIR resources als gewone Python dicts (JSON-serialiseerbaar).
Dit werkt direct met de HiX FHIR REST API.
"""

from datetime import datetime
from loguru import logger


def map_to_fhir(extracted: dict) -> dict:
    """
    Zet geëxtraheerde patiëntdata om naar een FHIR R4 Bundle.

    Een FHIR Bundle is een container voor meerdere resources —
    zo kunnen we alles in één API-call naar HiX sturen.

    Parameters:
        extracted: dict — output van extractor.extract_patient_data()

    Returns:
        dict — een FHIR R4 Bundle (JSON-serialiseerbaar)
    """
    patient = extracted.get("patient", {}) or {}
    medisch = extracted.get("medisch", {}) or {}

    bundle_entries = []

    # ── Patiënt resource ───────────────────────────────────────────────────────
    patient_resource = _build_patient_resource(patient)
    patient_id = patient_resource["id"]
    bundle_entries.append({
        "resource": patient_resource,
        "request": {
            "method": "PUT",                          # PUT = aanmaken of updaten
            "url": f"Patient/{patient_id}",
        },
    })

    # ── Diagnoses (Condition resources) ────────────────────────────────────────
    hoofddiagnose = medisch.get("hoofddiagnose")
    if hoofddiagnose:
        condition = _build_condition_resource(hoofddiagnose, patient_id, is_primary=True)
        bundle_entries.append({
            "resource": condition,
            "request": {"method": "POST", "url": "Condition"},
        })

    for nevendiagnose in medisch.get("nevendiagnoses", []) or []:
        condition = _build_condition_resource(nevendiagnose, patient_id, is_primary=False)
        bundle_entries.append({
            "resource": condition,
            "request": {"method": "POST", "url": "Condition"},
        })

    # ── Medicatie (MedicationStatement resources) ──────────────────────────────
    for medicijn in medisch.get("medicatie", []) or []:
        med_statement = _build_medication_statement(medicijn, patient_id)
        bundle_entries.append({
            "resource": med_statement,
            "request": {"method": "POST", "url": "MedicationStatement"},
        })

    # ── Bundle samenstellen ────────────────────────────────────────────────────
    bundle = {
        "resourceType": "Bundle",
        "type": "transaction",                        # Alles of niets — atomisch
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "entry": bundle_entries,
    }

    logger.success(
        f"  FHIR Bundle aangemaakt: "
        f"1 Patient, "
        f"{len([e for e in bundle_entries if e['resource']['resourceType'] == 'Condition'])} Condition(s), "
        f"{len([e for e in bundle_entries if e['resource']['resourceType'] == 'MedicationStatement'])} MedicationStatement(s)"
    )

    return bundle


def _build_patient_resource(patient: dict) -> dict:
    """Bouw een FHIR R4 Patient resource."""

    # Gebruik BSN als unieke identifier, anders een tijdelijke ID
    bsn = patient.get("bsn")
    patient_id = bsn if bsn else f"onbekend-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"

    resource = {
        "resourceType": "Patient",
        "id": patient_id,
        "meta": {
            "profile": ["http://nictiz.nl/fhir/StructureDefinition/nl-core-Patient"],
        },
    }

    # BSN als identifier (officieel NL-profiel)
    if bsn:
        resource["identifier"] = [
            {
                "use": "official",
                "system": "http://fhir.nl/fhir/NamingSystem/bsn",  # NL BSN naamgevingssysteem
                "value": bsn,
            }
        ]

    # Naam
    achternaam = patient.get("achternaam")
    voornaam = patient.get("voornaam")
    if achternaam or voornaam:
        resource["name"] = [
            {
                "use": "official",
                "family": achternaam or "",
                "given": [voornaam] if voornaam else [],
            }
        ]

    # Geboortedatum (FHIR verwacht YYYY-MM-DD)
    geboortedatum = patient.get("geboortedatum")
    if geboortedatum:
        resource["birthDate"] = _convert_date_to_fhir(geboortedatum)

    # Geslacht
    geslacht_map = {"M": "male", "V": "female"}
    geslacht = patient.get("geslacht")
    if geslacht in geslacht_map:
        resource["gender"] = geslacht_map[geslacht]

    # Adres
    adres = patient.get("adres")
    if adres or patient.get("woonplaats"):
        resource["address"] = [
            {
                "use": "home",
                "line": [adres] if adres else [],
                "postalCode": patient.get("postcode") or "",
                "city": patient.get("woonplaats") or "",
                "country": "NL",
            }
        ]

    return resource


def _build_condition_resource(diagnose: str, patient_id: str, is_primary: bool) -> dict:
    """Bouw een FHIR R4 Condition resource voor een diagnose."""
    return {
        "resourceType": "Condition",
        "clinicalStatus": {
            "coding": [
                {
                    "system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
                    "code": "active",
                }
            ]
        },
        "category": [
            {
                "coding": [
                    {
                        # Onderscheid primaire vs. nevendiagnose
                        "system": "http://terminology.hl7.org/CodeSystem/condition-category",
                        "code": "encounter-diagnosis" if is_primary else "problem-list-item",
                    }
                ]
            }
        ],
        "code": {
            # We hebben nog geen SNOMED/ICD-10 code — alleen de tekst
            # In een vervolgstap kan dit worden omgezet naar gecodeerde diagnoses
            "text": diagnose,
        },
        "subject": {
            "reference": f"Patient/{patient_id}",
        },
        "recordedDate": datetime.utcnow().strftime("%Y-%m-%d"),
    }


def _build_medication_statement(medicijn: dict, patient_id: str) -> dict:
    """Bouw een FHIR R4 MedicationStatement resource."""
    return {
        "resourceType": "MedicationStatement",
        "status": "active",
        "medicationCodeableConcept": {
            # Zonder G-standaard/ATC code — alleen de tekst uit het document
            "text": medicijn.get("naam", "Onbekend"),
        },
        "subject": {
            "reference": f"Patient/{patient_id}",
        },
        "dosage": [
            {
                "text": " ".join(filter(None, [
                    medicijn.get("dosering"),
                    medicijn.get("frequentie"),
                ])) or None,
            }
        ] if medicijn.get("dosering") or medicijn.get("frequentie") else [],
    }


def _convert_date_to_fhir(date_str: str) -> str:
    """
    Converteer datum van DD-MM-YYYY (ons formaat) naar YYYY-MM-DD (FHIR formaat).
    """
    try:
        dt = datetime.strptime(date_str, "%d-%m-%Y")
        return dt.strftime("%Y-%m-%d")
    except ValueError:
        # Als de datum al in FHIR formaat staat of onbekend formaat, geef terug as-is
        return date_str
