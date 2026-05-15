"""
pipeline/extractor.py
---------------------
Stap 2: Semantische data-extractie met een lokale LLM via Ollama.
"""

from __future__ import annotations

import copy
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
    EXTRACT_NIET_GEEXTRAHEERD,
)


EXTRACTION_SCHEMA: dict[str, Any] = {
    "patient": {
        "achternaam": "string | null — kijk in 'Betreft:', adresblok, aanhef",
        "voornaam": "string | null — kijk in 'Betreft:', adresblok; initialen zijn geen voornaam",
        "geboortedatum": "string in formaat DD-MM-YYYY | null — kijk naar 'Geb. datum:', 'geboren', leeftijdsvermelding",
        "bsn": "9-cijferig burgerservicenummer als string | null — kijk naar 'BSN:', 'Pat.nr./BSN:' (alleen exact 9 cijfers)",
        "geslacht": "M of V of null — kijk naar 'Dhr.'/'Mevr.', 'hij'/'zij'/'haar' in lopende tekst, voornaam",
        "adres": "string | null — straatnaam + huisnummer; kijk in adresblok na 'Adres:'",
        "postcode": "string | null — formaat 1234 AB; kijk in adresblok",
        "woonplaats": "string | null — kijk in adresblok na postcode",
    },
    "document": {
        "type": "bijv. ontslagbrief / verwijsbrief / polikliniekbrief | null — kijk bij 'Type bericht:' of afleidbaar uit context",
        "ziekenhuis_van_herkomst": "naam van het ziekenhuis dat het document stuurde | null — kijk in briefhoofd, afzenderadres, website",
        "afdeling": "string | null — kijk bij 'afdeling:', briefhoofd, ondertekening",
    },
    "medisch": {
        "hoofddiagnose": "string | null — primaire reden van opname/verwijzing; kijk onder 'Conclusie', 'Diagnose', eerste alinea",
        "nevendiagnoses": ["huidige actieve nevendiagnoses en comorbiditeiten; kijk onder 'Conclusie' na punt 1, 'bekend met'"],
        "voorgeschiedenis": {
            "cardiaal": ["alleen items die letterlijk onder 'Voorgeschiedenis' → 'Cardiaal:' staan; niet afleiden uit anamnese of diagnoses elders in de brief; formaat per item: 'JJJJ diagnose' of 'JJJJ-MM diagnose', bijv. '2019 CABG', '2021-03 PCI RDA'; als jaar onbekend: zet item zonder jaar"],
            "overig":   ["alleen items die letterlijk onder 'Voorgeschiedenis' → 'Overige:' of 'Overig:' staan; niet afleiden uit anamnese of diagnoses elders in de brief; zelfde formaat: 'JJJJ diagnose' of 'JJJJ-MM diagnose'"],
        },
        "medicatie": [
            {
                "naam": "string — alleen de generieke stofnaam of merknaam van het medicijn, zonder toedieningsvorm, sterkte of route; bijv. 'metoprolol', 'acetylsalicylzuur', 'brinzolamide'; extraheer ALLE medicijnen uit de sectie 'Actieve medicatie', ongeacht indicatie of toedieningsroute (inclusief oogdruppels, inhalatoren, niet-cardiale middelen); sla niets over",
                "dosering": "string | null — sterkte, bijv. '40mg'",
                "frequentie": "string | null — hoe vaak, bijv. '1 x per dag', '2 x per dag', 'zo nodig'; bij 'zo nodig': schrijf alleen 'zo nodig', geen aanvullende tekst; laat toedieningsroute (oraal, ogen, sc) weg",
            }
        ],
        "allergieën": ["alle vermelde allergieën/intoleranties; kijk onder 'Allergieën', 'CAVE'; bij 'Geen' schrijf []"],
        "behandelend_arts": "naam van de arts die de patiënt behandelt | null — kijk in ondertekening na 'Namens', briefhoofd",
        "verwijzend_arts": "naam van de verwijzende arts | null — kijk in ondertekening (afzender), 'Met vriendelijke groet', 'Met collegiale groet'",
        "verwijzend_ziekenhuis": "naam van het ziekenhuis dat de patiënt verwijst | null — zelfde bron als document.ziekenhuis_van_herkomst; kijk in briefhoofd, afzenderadres, website",
        "opname_type": "'poliklinisch' | 'klinisch' | null — poliklinisch als patiënt op de poli werd gezien, klinisch als patiënt opgenomen was; kijk naar 'opname', 'poliklinisch', 'klinisch', context van de brief",
        "reden_van_verwijzing": "string | null — de specifieke reden waarom deze patiënt wordt aangemeld bij het hartteam of verwezen naar LUMC; zoek primair in de secties 'Beleid' en 'Conclusie' naar termen als 'aanmelden hartteam', 'aanbieden in LUMC', 'aanbieden te LUMC', 'voorstel', gevolgd door een behandeling; neem het behandelvoorstel op in de waarde, bijv. 'Aanmelden hartteam voorstel CABG', 'Aanbieden LUMC PCI RDA', 'Voorstel conservatief beleid'; gebruik NIET de algemene opnamereden of klacht bij binnenkomst",
    },
    "anamnese": {
        "pijn_op_de_borst": "true | false | null — pob/AP-klachten/angina pectoris/druk op de borst/pijn op de borst; null als niet vermeld",
        "pob_bij_inspanning": "true | false | null — alleen true als pob expliciet bij inspanning/lopen/wandelen wordt beschreven; null als niet vermeld",
        "pob_in_rust": "true | false | null — alleen true als pob expliciet in rust of nacht wordt beschreven; null als niet vermeld",
        "dyspnoe": "true | false | null — kortademigheid/kortadem/naar adem happen/dyspnoe; null als niet vermeld",
        "dyspnoe_bij_inspanning": "true | false | null — true als dyspnoe expliciet bij inspanning staat, OF als dyspnoe wordt vermeld in een zin die als geheel gaat over inspanningsklachten (bijv. 'bij wandelen druk op de borst en kortademigheidsklachten' — de 'bij wandelen' geldt dan voor alle klachten in die zin); null als niet vermeld",
        "palpitaties": "true | false | null — hartkloppingen/palpitaties; null als niet vermeld",
        "syncope": "true | false | null — wegrakingen/bewustzijnsverlies/flauwvallen; null als niet vermeld",
        "oedeem": "true | false | null — vocht vasthouden/zwelling benen/oedeem; null als niet vermeld",
        "roken": "string | null — gebruik uitsluitend: 'nooit' (nooit gerookt), 'gestopt [jaar]' (bijv. 'gestopt 2004', inclusief pack years indien vermeld, bijv. 'gestopt 2004, 20 pack years'), of 'ja' (actief roker, inclusief hoeveelheid indien vermeld, bijv. 'ja, 10 sigaretten per dag, 15 pack years'); maak geen aanname als de tekst onduidelijk is: gebruik dan null; 'rookt niet' zonder verdere context mag als 'nooit' worden genoteerd alleen als er geen aanwijzing is voor stoppen",
        "alcohol": "string | null — bijv. 'sporadisch', 'nooit', '2 eenheden per dag'; null als niet vermeld",
        "drugs": "string | null — bijv. 'nooit', 'ja cannabis'; null als niet vermeld",
        "familieanamnese": "string | null — verbatim, bijv. 'geen hvz', 'vader MI op 52-jarige leeftijd'; null als niet vermeld",
        "overige": ["overige klinisch relevante anamnesevermeldingen die niet passen in bovenstaande velden, verbatim; bijv. gewichtsverlies, koorts, duizeligheid"],
    },
    "diagnostiek": {
        "TTE": {
            "datum": "string in formaat DD-MM-YYYY | null — kijk naar 'Echo cor d.d.', 'TTE d.d.', 'Transthoracaal'; als het document alleen 'echo cor' vermeldt zonder typering dan is het TTE",
            "LV": {
                "dimensies": "string | null — bijv. 'normaal', 'niet gedilateerd', 'gedilateerd'",
                "LVIDd": "string | null — bijv. '57 mm'",
                "hypertrofie": "string | null — bijv. 'geen hypertrofie', 'licht hypertrofisch'",
                "systolische_functie": "string | null — bijv. 'goede systolische functie', 'redelijk'",
                "EF": "string | null — bijv. '63%', '45-50%'",
                "diastolische_functie": "string | null — bijv. 'normaal', 'gestoord graad 2'",
                "E_e_prime": "string | null — bijv. '11'",
                "E_A_verhouding": "string | null — bijv. 'E>A'",
                "S_D_verhouding": "string | null — bijv. 'nbtm'",
                "RWBS": "string | null — bijv. 'geen rwbs', 'Hypokinesie ant lat en ant'",
                "overige": ["overige LV-bevindingen verbatim, bijv. 'False tendon'"],
            },
            "RV": {
                "dimensies": "string | null",
                "functie": "string | null — bijv. 'goede functie', 'normaal gelet op de TAPSE'",
                "TAPSE": "string | null — bijv. '30 mm', '21 mm'",
                "S_prime": "string | null — bijv. '0.17 m/s'",
                "overige": ["overige RV-bevindingen verbatim"],
            },
            "LA": {
                "dimensies": "string | null",
                "LAVI": "string | null — bijv. '31 ml/m2'",
                "LAESV_index": "string | null — bijv. '33 ml/m2'",
                "volume": "string | null — bijv. '38 ml/m2'",
                "overige": ["overige LA-bevindingen verbatim"],
            },
            "RA": {
                "dimensies": "string | null",
                "RAVI": "string | null — bijv. '30 ml/m2'",
                "RAESV": "string | null — bijv. '49 ml'",
                "area": "string | null — bijv. '18 cm2'",
                "overige": ["overige RA-bevindingen verbatim"],
            },
            "kleppen": {
                "AOV": {
                    "morfologie": "string | null — bijv. 'tricuspide', 'bicuspide'",
                    "opening": "string | null — bijv. 'opent goed', 'sclerotisch met goede opening'",
                    "gradienten": "string | null — bijv. 'normale gradienten', 'matige gradienten'",
                    "insufficientie": "string | null — bijv. 'geen', 'geringe', 'matige'",
                    "AI_P1_2t": "string | null — bijv. '440,0 msec'",
                    "AI_end_d_velocity": "string | null — bijv. '14,3 cm/sec'",
                    "overige": ["overige AOV-bevindingen verbatim"],
                },
                "MV": {
                    "morfologie": "string | null",
                    "mean_PG": "string | null — bijv. '1,91 mmHg'",
                    "insufficientie": "string | null — bijv. 'geen', 'geringe'",
                    "calcificatie_annulus": "string | null — bijv. 'geringe calcificatie'",
                    "overige": ["overige MV-bevindingen verbatim"],
                },
                "TV": {
                    "insufficientie": "string | null — bijv. 'geen', 'geringe'",
                    "Ti_Vmax": "string | null — bijv. 'nbtm'",
                    "max_PG": "string | null — bijv. '22 mmHg'",
                    "overige": ["overige TV-bevindingen verbatim"],
                },
                "PV": {
                    "insufficientie": "string | null — bijv. 'geen', 'geringe'",
                    "overige": ["overige PV-bevindingen verbatim"],
                },
            },
            "aorta": {
                "AO_dimensies": "string | null — bijv. 'normaal'",
                "aorta_ascendens": "string | null — bijv. '56 mm'",
                "AO_root": "string | null — bijv. '40 mm'",
                "AO_boog": "string | null — bijv. '33 mm'",
                "AO_abdominalis": "string | null — bijv. '24 mm'",
            },
            "VCI": {
                "collaps": "string | null — bijv. 'goede collaps', 'normaal'",
                "geschatte_CVD": "string | null — bijv. '0-5 mmHg'",
                "sPAP": "string | null — bijv. '20 mmHg'",
                "overige": ["overige VCI-bevindingen verbatim"],
            },
            "PHT": {
                "sPAP": "string | null",
                "secundaire_aanwijzingen": "string | null — bijv. 'geen secundaire aanwijzingen PHT'",
            },
            "PE": "string | null — bijv. 'geen'",
            "overige_bevindingen": ["overige bevindingen die niet in bovenstaande velden passen, verbatim"],
            "conclusie": ["één regel per conclusiepunt, verbatim uit het verslag"],
        },
        "TEE": {
            "datum": "string in formaat DD-MM-YYYY | null — kijk naar 'TEE d.d.', 'Transoesophageaal', 'slokdarmecho'; null als niet aanwezig in document",
            "LV": {
                "dimensies": "string | null",
                "LVIDd": "string | null",
                "hypertrofie": "string | null",
                "systolische_functie": "string | null",
                "EF": "string | null",
                "diastolische_functie": "string | null",
                "E_e_prime": "string | null",
                "E_A_verhouding": "string | null",
                "S_D_verhouding": "string | null",
                "RWBS": "string | null",
                "overige": ["overige LV-bevindingen verbatim"],
            },
            "RV": {
                "dimensies": "string | null",
                "functie": "string | null",
                "TAPSE": "string | null",
                "S_prime": "string | null",
                "overige": ["overige RV-bevindingen verbatim"],
            },
            "LA": {
                "dimensies": "string | null",
                "LAVI": "string | null",
                "LAESV_index": "string | null",
                "volume": "string | null",
                "overige": ["overige LA-bevindingen verbatim"],
            },
            "RA": {
                "dimensies": "string | null",
                "RAVI": "string | null",
                "RAESV": "string | null",
                "area": "string | null",
                "overige": ["overige RA-bevindingen verbatim"],
            },
            "kleppen": {
                "AOV": {
                    "morfologie": "string | null",
                    "opening": "string | null",
                    "gradienten": "string | null",
                    "insufficientie": "string | null",
                    "AI_P1_2t": "string | null",
                    "AI_end_d_velocity": "string | null",
                    "overige": ["overige AOV-bevindingen verbatim"],
                },
                "MV": {
                    "morfologie": "string | null",
                    "mean_PG": "string | null",
                    "insufficientie": "string | null",
                    "calcificatie_annulus": "string | null",
                    "overige": ["overige MV-bevindingen verbatim"],
                },
                "TV": {
                    "insufficientie": "string | null",
                    "Ti_Vmax": "string | null",
                    "max_PG": "string | null",
                    "overige": ["overige TV-bevindingen verbatim"],
                },
                "PV": {
                    "insufficientie": "string | null",
                    "overige": ["overige PV-bevindingen verbatim"],
                },
            },
            "aorta": {
                "AO_dimensies": "string | null",
                "aorta_ascendens": "string | null",
                "AO_root": "string | null",
                "AO_boog": "string | null",
                "AO_abdominalis": "string | null",
            },
            "VCI": {
                "collaps": "string | null",
                "geschatte_CVD": "string | null",
                "sPAP": "string | null",
                "overige": ["overige VCI-bevindingen verbatim"],
            },
            "PHT": {
                "sPAP": "string | null",
                "secundaire_aanwijzingen": "string | null",
            },
            "PE": "string | null",
            "overige_bevindingen": ["overige bevindingen die niet in bovenstaande velden passen, verbatim"],
            "conclusie": ["één regel per conclusiepunt, verbatim uit het verslag"],
        },
        "cag": {
            "datum": "string | null — kijk in koptekst ('CAG 29-4', 'CAG 06-03-2026', 'CAG d.d. 30-04-2026')",
            "dominantie": "string | null — bijv. 'rechts-dominant', 'links-dominant', 'co-dominant'; dominantie wordt bepaald door welk vat de PDA/RPD levert",
            "natieve_coronairen": {
                "RCA": {
                    "bevinding": "string | null — volledige verbatim beschrijving; herkent ook: rechter kransslagader, right coronary artery, arteria coronaria dextra; inclusief zijtakken: conustak (CB), SA-knooptak (SNA), acute marginaaltak (AM), posterior descenderende tak (PDA/RPD) bij rechter dominantie, posterolaterale tak (PLV/PL), AV-knooptak (AVN)",
                    "stenose_locatie": ["locaties van stenosen verbatim, bijv. 'proximaal', 'distaal in RPL'"],
                    "stenose_graad": ["percentages verbatim, bijv. '90%', '80%'"],
                    "overige": ["overige bevindingen voor dit vat verbatim"],
                },
                "LM": {
                    "bevinding": "string | null — volledige verbatim beschrijving; herkent ook: linker hoofdstam, left main, LMCA, left main coronary artery, truncus communis sinister, LCA",
                    "stenose_locatie": [],
                    "stenose_graad": [],
                    "overige": [],
                },
                "RDA": {
                    "bevinding": "string | null — volledige verbatim beschrijving; herkent ook: ramus descendens anterior, LAD, left anterior descending, voorste neergaande tak, LVA; inclusief zijtakken: diagonaaltak (D1/D2/D3), septaalperforatoren (S1/S2)",
                    "stenose_locatie": ["locaties van stenosen verbatim"],
                    "stenose_graad": ["percentages verbatim"],
                    "overige": ["overige bevindingen voor dit vat verbatim"],
                },
                "RCX": {
                    "bevinding": "string | null — volledige verbatim beschrijving; herkent ook: ramus circumflexus, LCx, Cx, left circumflex, ombuigende tak, LCA-Cx; inclusief zijtakken: obtuse marginaaltak (OM1/OM2), ramus intermedius (RI), posterolaterale tak (PLV), posterior descenderende tak (PDA/RPD) bij linker dominantie",
                    "stenose_locatie": ["locaties van stenosen verbatim"],
                    "stenose_graad": ["percentages verbatim"],
                    "overige": ["overige bevindingen voor dit vat verbatim"],
                },
            },
            "collateralen": "string | null — bijv. 'afwezig', 'aanwezig'",
            "hemostase": "string | null — bijv. 'TR band', 'angioseal'",
            "complicaties": "string | null — bijv. 'geen', 'hematoom'",
            "conclusie": ["één regel per conclusiepunt, verbatim; als er geen gestructureerd CAG-verslag aanwezig is maar CAG-bevindingen staan wel in de secties 'Beloop', 'Conclusie', of 'Overweging', neem deze dan over; voeg in dat geval een veld 'bron' toe met waarde 'beloop' of 'cag_verslag' om de herkomst aan te geven"],
            "beleid": ["één regel per beleidspunt, verbatim"],
            "overige_bevindingen": ["overige bevindingen die niet passen in bovenstaande velden, verbatim"],
        },
        "ecg": "verbatim bevindingen van ECG | null — kijk naar 'ECG'",
        "beeldvorming": {
            "pet_ct": {
                "datum": "string | null — kijk in koptekst ('PET-CT 2026-04', 'PET april 2026')",
                "regels": ["alle regels van het PET-CT verslag verbatim, één string per regel; als de sectie is opgebouwd met genummerde regels (1., 2., 3., 4. etc.), neem dan ALLE genummerde regels op in dit veld zonder uitzondering — ook als een regel een CT-component beschrijft die onderdeel is van de PET-CT; een regel hoort hier thuis als hij onder het kopje PET-CT staat, ongeacht de inhoud; alleen een volledig losstaand CT-onderzoek met een eigen kopje hoort bij ct_coronairen of ct_thorax"],
            },
            "ct_coronairen": {
                "datum": "string | null",
                "regels": ["alle regels van het CT coronairen verslag verbatim, één string per regel of punt"],
            },
            "ct_thorax": {
                "datum": "string | null",
                "regels": ["alle regels van het CT thorax verslag verbatim, één string per regel of punt"],
            },
            "mri_hart": {
                "datum": "string | null — alleen invullen als er een MRI hart verslag aanwezig is in de brief met een eigen sectie of expliciete datum; vermeld uit de voorgeschiedenis bekende MRI-bevindingen NIET hier; die staan al in voorgeschiedenis.cardiaal",
                "regels": ["alle regels van het MRI hart verslag verbatim, één string per regel of punt"],
            },
            "overige_beeldvorming": [
                {
                    "modaliteit": "string — naam van de modaliteit verbatim, bijv. 'CT abdomen', 'X-thorax'",
                    "datum": "string | null",
                    "regels": ["alle regels verbatim"],
                }
            ],
        },
        "overige_onderzoeken": [{"naam": "string", "datum": "string | null", "conclusie": "verbatim string | null", "status": "'uitgevoerd' | 'aangevraagd' | null — gebruik 'aangevraagd' als het onderzoek is aangevraagd maar de uitslag nog niet beschikbaar is in de brief; gebruik 'uitgevoerd' als er een uitslag of conclusie aanwezig is; null als onduidelijk"}],
    },
    "laboratorium": {
        "datum": "string | null — afnamedatum van het labblok, formaat DD-MM-YYYY; kijk naar expliciete datumvermelding boven het labblok ('Laboratorium 10-4-2026', 'Afnamedatum: 29-04-2026'); gebruik niet de datum van de brief zelf",
        "hb": "string | null — hemoglobine; alleen numerieke waarde met eenheid, zonder vlaggen zoals (L) of (H); bijv. '7.3 mmol/L' niet '7.3 (L) mmol/L'; bij '<' of '>' prefix behouden",
        "ht": "string | null — hematocriet; alleen numerieke waarde met eenheid, zonder vlaggen zoals (L) of (H); bijv. '0.36 L/L'; bij '<' of '>' prefix behouden",
        "leukocyten": "string | null — alleen numerieke waarde met eenheid, zonder vlaggen zoals (L) of (H); bijv. '9.1 *10^9/L'; bij '<' of '>' prefix behouden",
        "trombocyten": "string | null — alleen numerieke waarde met eenheid, zonder vlaggen zoals (L) of (H); bijv. '182 *10^9/L'; bij '<' of '>' prefix behouden",
        "natrium": "string | null — alleen numerieke waarde met eenheid, zonder vlaggen zoals (L) of (H); bijv. '133 mmol/L'; bij '<' of '>' prefix behouden",
        "kalium": "string | null — alleen numerieke waarde met eenheid, zonder vlaggen zoals (L) of (H); bijv. '3.8 mmol/L'; bij '<' of '>' prefix behouden",
        "ureum": "string | null — alleen numerieke waarde met eenheid, zonder vlaggen zoals (L) of (H); bijv. '8.8 mmol/L'; bij '<' of '>' prefix behouden",
        "kreatinine": "string | null — alleen numerieke waarde met eenheid, zonder vlaggen zoals (L) of (H); bijv. '103 umol/L'; bij '<' of '>' prefix behouden",
        "egfr": "string | null — alleen numerieke waarde met eenheid, zonder vlaggen zoals (L) of (H); bijv. '59 mL/min/1.73m2'; bij '<' of '>' prefix behouden",
        "glucose": "string | null — alleen numerieke waarde met eenheid, zonder vlaggen zoals (L) of (H); bijv. '10.1 mmol/L'; bij '<' of '>' prefix behouden",
        "crp": "string | null — alleen numerieke waarde met eenheid, zonder vlaggen zoals (L) of (H); bijv. '28 mg/L'; bij '<' of '>' prefix behouden",
        "hs_troponine": "string | null — hsTroponine-I of hsTroponine-T; alleen numerieke waarde met eenheid, zonder vlaggen zoals (L) of (H); bijv. '494.9 ng/L'; bij '<' of '>' prefix behouden",
        "ck": "string | null — creatinekinase; alleen numerieke waarde met eenheid, zonder vlaggen zoals (L) of (H); bijv. '114 U/L'; bij '<' of '>' prefix behouden",
        "ld": "string | null — lactaatdehydrogenase; alleen numerieke waarde met eenheid, zonder vlaggen zoals (L) of (H); bijv. '261 U/L'; bij '<' of '>' prefix behouden",
        "asat": "string | null — alleen numerieke waarde met eenheid, zonder vlaggen zoals (L) of (H); bijv. '28 U/L'; bij '<' of '>' prefix behouden",
        "alat": "string | null — alleen numerieke waarde met eenheid, zonder vlaggen zoals (L) of (H); bijv. '25 U/L'; bij '<' of '>' prefix behouden",
        "cholesterol": "string | null — totaal cholesterol; alleen numerieke waarde met eenheid, zonder vlaggen zoals (L) of (H); bijv. '3.2 mmol/L'; bij '<' of '>' prefix behouden",
        "hdl": "string | null — HDL cholesterol; alleen numerieke waarde met eenheid, zonder vlaggen zoals (L) of (H); bijv. '1.57 mmol/L'; bij '<' of '>' prefix behouden",
        "ldl": "string | null — LDL cholesterol; alleen numerieke waarde met eenheid, zonder vlaggen zoals (L) of (H); bijv. '1.36 mmol/L'; bij '<' of '>' prefix behouden",
        "triglyceriden": "string | null — alleen numerieke waarde met eenheid, zonder vlaggen zoals (L) of (H); bijv. '0.45 mmol/L'; bij '<' of '>' prefix behouden",
        "lpa": "string | null — Lp(a); alleen numerieke waarde met eenheid, zonder vlaggen zoals (L) of (H); bijv. '174 g/L'; bij '<' of '>' prefix behouden",
        "tsh": "string | null — alleen numerieke waarde met eenheid, zonder vlaggen zoals (L) of (H); bijv. '1.08 mU/L'; bij '<' of '>' prefix behouden",
        "nt_pro_bnp": "string | null — NTproBNP of NT-proBNP; alleen numerieke waarde met eenheid, zonder vlaggen zoals (L) of (H); bijv. '109 ng/L'; bij '<' of '>' prefix behouden",
        "calcium": "string | null — alleen numerieke waarde met eenheid, zonder vlaggen zoals (L) of (H); bijv. '2.01 mmol/L'; bij '<' of '>' prefix behouden",
        "overige": ["overige labwaarden die niet passen in bovenstaande velden, verbatim als 'naam: waarde eenheid'"],
    },
    "extractie_metadata": {
        "ontbrekende_velden": ["lijst van velden die je na grondig zoeken écht niet kon vinden in de tekst"],
        "opmerkingen": "string | null — twijfelgevallen of bijzonderheden over dit document",
        "niet_geextraheerd": ["verbatim snippets of omschrijvingen van inhoud die wel in de brontekst stond maar in geen enkel ander veld paste; bijv. 'NYHA klasse II', 'stress-echo negatief', 'Holter: geen ritmestoornissen'; laat leeg als alles is ondergebracht"],
    },
}


EXTRACTION_TEMPLATE: dict[str, Any] = {
    "patient": {
        "achternaam": None,
        "voornaam": None,
        "geboortedatum": None,
        "bsn": None,
        "geslacht": None,
        "adres": None,
        "postcode": None,
        "woonplaats": None,
    },
    "document": {
        "type": None,
        "ziekenhuis_van_herkomst": None,
        "afdeling": None,
    },
    "medisch": {
        "hoofddiagnose": None,
        "nevendiagnoses": [],
        "voorgeschiedenis": {"cardiaal": [], "overig": []},
        "medicatie": [],
        "allergieën": [],
        "behandelend_arts": None,
        "verwijzend_arts": None,
        "verwijzend_ziekenhuis": None,
        "opname_type": None,
        "reden_van_verwijzing": None,
    },
    "anamnese": {
        "pijn_op_de_borst": None,
        "pob_bij_inspanning": None,
        "pob_in_rust": None,
        "dyspnoe": None,
        "dyspnoe_bij_inspanning": None,
        "palpitaties": None,
        "syncope": None,
        "oedeem": None,
        "roken": None,
        "alcohol": None,
        "drugs": None,
        "familieanamnese": None,
        "overige": [],
    },
    "diagnostiek": {
        "TTE": {
            "datum": None,
            "LV": {"dimensies": None, "LVIDd": None, "hypertrofie": None, "systolische_functie": None, "EF": None, "diastolische_functie": None, "E_e_prime": None, "E_A_verhouding": None, "S_D_verhouding": None, "RWBS": None, "overige": []},
            "RV": {"dimensies": None, "functie": None, "TAPSE": None, "S_prime": None, "overige": []},
            "LA": {"dimensies": None, "LAVI": None, "LAESV_index": None, "volume": None, "overige": []},
            "RA": {"dimensies": None, "RAVI": None, "RAESV": None, "area": None, "overige": []},
            "kleppen": {
                "AOV": {"morfologie": None, "opening": None, "gradienten": None, "insufficientie": None, "AI_P1_2t": None, "AI_end_d_velocity": None, "overige": []},
                "MV": {"morfologie": None, "mean_PG": None, "insufficientie": None, "calcificatie_annulus": None, "overige": []},
                "TV": {"insufficientie": None, "Ti_Vmax": None, "max_PG": None, "overige": []},
                "PV": {"insufficientie": None, "overige": []},
            },
            "aorta": {"AO_dimensies": None, "aorta_ascendens": None, "AO_root": None, "AO_boog": None, "AO_abdominalis": None},
            "VCI": {"collaps": None, "geschatte_CVD": None, "sPAP": None, "overige": []},
            "PHT": {"sPAP": None, "secundaire_aanwijzingen": None},
            "PE": None,
            "overige_bevindingen": [],
            "conclusie": [],
        },
        "TEE": {
            "datum": None,
            "LV": {"dimensies": None, "LVIDd": None, "hypertrofie": None, "systolische_functie": None, "EF": None, "diastolische_functie": None, "E_e_prime": None, "E_A_verhouding": None, "S_D_verhouding": None, "RWBS": None, "overige": []},
            "RV": {"dimensies": None, "functie": None, "TAPSE": None, "S_prime": None, "overige": []},
            "LA": {"dimensies": None, "LAVI": None, "LAESV_index": None, "volume": None, "overige": []},
            "RA": {"dimensies": None, "RAVI": None, "RAESV": None, "area": None, "overige": []},
            "kleppen": {
                "AOV": {"morfologie": None, "opening": None, "gradienten": None, "insufficientie": None, "AI_P1_2t": None, "AI_end_d_velocity": None, "overige": []},
                "MV": {"morfologie": None, "mean_PG": None, "insufficientie": None, "calcificatie_annulus": None, "overige": []},
                "TV": {"insufficientie": None, "Ti_Vmax": None, "max_PG": None, "overige": []},
                "PV": {"insufficientie": None, "overige": []},
            },
            "aorta": {"AO_dimensies": None, "aorta_ascendens": None, "AO_root": None, "AO_boog": None, "AO_abdominalis": None},
            "VCI": {"collaps": None, "geschatte_CVD": None, "sPAP": None, "overige": []},
            "PHT": {"sPAP": None, "secundaire_aanwijzingen": None},
            "PE": None,
            "overige_bevindingen": [],
            "conclusie": [],
        },
        "cag": {
            "datum": None,
            "dominantie": None,
            "natieve_coronairen": {
                "RCA": {"bevinding": None, "stenose_locatie": [], "stenose_graad": [], "overige": []},
                "LM":  {"bevinding": None, "stenose_locatie": [], "stenose_graad": [], "overige": []},
                "RDA": {"bevinding": None, "stenose_locatie": [], "stenose_graad": [], "overige": []},
                "RCX": {"bevinding": None, "stenose_locatie": [], "stenose_graad": [], "overige": []},
            },
            "collateralen": None,
            "hemostase": None,
            "complicaties": None,
            "conclusie": [],
            "beleid": [],
            "overige_bevindingen": [],
        },
        "ecg": None,
        "beeldvorming": {
            "pet_ct":        {"datum": None, "regels": []},
            "ct_coronairen": {"datum": None, "regels": []},
            "ct_thorax":     {"datum": None, "regels": []},
            "mri_hart":      {"datum": None, "regels": []},
            "overige_beeldvorming": [],
        },
        "overige_onderzoeken": [],
    },
    "laboratorium": {
        "datum": None,
        "hb": None,
        "ht": None,
        "leukocyten": None,
        "trombocyten": None,
        "natrium": None,
        "kalium": None,
        "ureum": None,
        "kreatinine": None,
        "egfr": None,
        "glucose": None,
        "crp": None,
        "hs_troponine": None,
        "ck": None,
        "ld": None,
        "asat": None,
        "alat": None,
        "cholesterol": None,
        "hdl": None,
        "ldl": None,
        "triglyceriden": None,
        "lpa": None,
        "tsh": None,
        "nt_pro_bnp": None,
        "calcium": None,
        "overige": [],
    },
    "extractie_metadata": {
        "ontbrekende_velden": [],
        "opmerkingen": None,
        "niet_geextraheerd": [],
    },
}


# JSON Schema voor Ollama structured output — dwingt structuur af op tokensniveau.
EXTRACTION_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "patient": {
            "type": "object",
            "properties": {
                "achternaam":         {"type": ["string", "null"]},
                "voornaam":           {"type": ["string", "null"]},
                "geboortedatum":      {"type": ["string", "null"]},
                "bsn":                {"type": ["string", "null"]},
                "geslacht":           {"type": ["string", "null"]},
                "adres":              {"type": ["string", "null"]},
                "postcode":           {"type": ["string", "null"]},
                "woonplaats":         {"type": ["string", "null"]},
            },
            "required": ["achternaam", "voornaam", "geboortedatum", "bsn", "geslacht", "adres", "postcode", "woonplaats"],
            "additionalProperties": False,
        },
        "document": {
            "type": "object",
            "properties": {
                "type":                   {"type": ["string", "null"]},
                "ziekenhuis_van_herkomst":{"type": ["string", "null"]},
                "afdeling":               {"type": ["string", "null"]},
            },
            "required": ["type", "ziekenhuis_van_herkomst", "afdeling"],
            "additionalProperties": False,
        },
        "medisch": {
            "type": "object",
            "properties": {
                "hoofddiagnose":      {"type": ["string", "null"]},
                "nevendiagnoses":     {"type": "array", "items": {"type": "string"}},
                "voorgeschiedenis": {
                    "type": "object",
                    "properties": {
                        "cardiaal": {"type": "array", "items": {"type": "string"}},
                        "overig":   {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["cardiaal", "overig"],
                    "additionalProperties": False,
                },
                "medicatie": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "naam":      {"type": "string"},
                            "dosering":  {"type": ["string", "null"]},
                            "frequentie":{"type": ["string", "null"]},
                        },
                        "required": ["naam", "dosering", "frequentie"],
                        "additionalProperties": False,
                    },
                },
                "allergieën":         {"type": "array", "items": {"type": "string"}},
                "behandelend_arts":        {"type": ["string", "null"]},
                "verwijzend_arts":         {"type": ["string", "null"]},
                "verwijzend_ziekenhuis":   {"type": ["string", "null"]},
                "opname_type":             {"type": ["string", "null"]},
                "reden_van_verwijzing": {"type": ["string", "null"]},
            },
            "required": ["hoofddiagnose", "nevendiagnoses", "voorgeschiedenis", "medicatie",
                         "allergieën", "behandelend_arts", "verwijzend_arts",
                         "verwijzend_ziekenhuis", "opname_type", "reden_van_verwijzing"],
            "additionalProperties": False,
        },
        "anamnese": {
            "type": "object",
            "properties": {
                "pijn_op_de_borst":       {"type": ["boolean", "null"]},
                "pob_bij_inspanning":     {"type": ["boolean", "null"]},
                "pob_in_rust":            {"type": ["boolean", "null"]},
                "dyspnoe":                {"type": ["boolean", "null"]},
                "dyspnoe_bij_inspanning": {"type": ["boolean", "null"]},
                "palpitaties":            {"type": ["boolean", "null"]},
                "syncope":                {"type": ["boolean", "null"]},
                "oedeem":                 {"type": ["boolean", "null"]},
                "roken":                  {"type": ["string", "null"]},
                "alcohol":                {"type": ["string", "null"]},
                "drugs":                  {"type": ["string", "null"]},
                "familieanamnese":        {"type": ["string", "null"]},
                "overige":                {"type": "array", "items": {"type": "string"}},
            },
            "required": [
                "pijn_op_de_borst", "pob_bij_inspanning", "pob_in_rust",
                "dyspnoe", "dyspnoe_bij_inspanning", "palpitaties",
                "syncope", "oedeem", "roken", "alcohol", "drugs",
                "familieanamnese", "overige"
            ],
            "additionalProperties": False,
        },
        "diagnostiek": {
            "type": "object",
            "properties": {
                "TTE": {
                    "type": ["object", "null"],
                    "properties": {
                        "datum": {"type": ["string", "null"]},
                        "LV": {"type": ["object", "null"], "properties": {"dimensies": {"type": ["string", "null"]}, "LVIDd": {"type": ["string", "null"]}, "hypertrofie": {"type": ["string", "null"]}, "systolische_functie": {"type": ["string", "null"]}, "EF": {"type": ["string", "null"]}, "diastolische_functie": {"type": ["string", "null"]}, "E_e_prime": {"type": ["string", "null"]}, "E_A_verhouding": {"type": ["string", "null"]}, "S_D_verhouding": {"type": ["string", "null"]}, "RWBS": {"type": ["string", "null"]}, "overige": {"type": "array", "items": {"type": "string"}}}, "required": ["dimensies", "LVIDd", "hypertrofie", "systolische_functie", "EF", "diastolische_functie", "E_e_prime", "E_A_verhouding", "S_D_verhouding", "RWBS", "overige"], "additionalProperties": False},
                        "RV": {"type": ["object", "null"], "properties": {"dimensies": {"type": ["string", "null"]}, "functie": {"type": ["string", "null"]}, "TAPSE": {"type": ["string", "null"]}, "S_prime": {"type": ["string", "null"]}, "overige": {"type": "array", "items": {"type": "string"}}}, "required": ["dimensies", "functie", "TAPSE", "S_prime", "overige"], "additionalProperties": False},
                        "LA": {"type": ["object", "null"], "properties": {"dimensies": {"type": ["string", "null"]}, "LAVI": {"type": ["string", "null"]}, "LAESV_index": {"type": ["string", "null"]}, "volume": {"type": ["string", "null"]}, "overige": {"type": "array", "items": {"type": "string"}}}, "required": ["dimensies", "LAVI", "LAESV_index", "volume", "overige"], "additionalProperties": False},
                        "RA": {"type": ["object", "null"], "properties": {"dimensies": {"type": ["string", "null"]}, "RAVI": {"type": ["string", "null"]}, "RAESV": {"type": ["string", "null"]}, "area": {"type": ["string", "null"]}, "overige": {"type": "array", "items": {"type": "string"}}}, "required": ["dimensies", "RAVI", "RAESV", "area", "overige"], "additionalProperties": False},
                        "kleppen": {
                            "type": ["object", "null"],
                            "properties": {
                                "AOV": {"type": ["object", "null"], "properties": {"morfologie": {"type": ["string", "null"]}, "opening": {"type": ["string", "null"]}, "gradienten": {"type": ["string", "null"]}, "insufficientie": {"type": ["string", "null"]}, "AI_P1_2t": {"type": ["string", "null"]}, "AI_end_d_velocity": {"type": ["string", "null"]}, "overige": {"type": "array", "items": {"type": "string"}}}, "required": ["morfologie", "opening", "gradienten", "insufficientie", "AI_P1_2t", "AI_end_d_velocity", "overige"], "additionalProperties": False},
                                "MV":  {"type": ["object", "null"], "properties": {"morfologie": {"type": ["string", "null"]}, "mean_PG": {"type": ["string", "null"]}, "insufficientie": {"type": ["string", "null"]}, "calcificatie_annulus": {"type": ["string", "null"]}, "overige": {"type": "array", "items": {"type": "string"}}}, "required": ["morfologie", "mean_PG", "insufficientie", "calcificatie_annulus", "overige"], "additionalProperties": False},
                                "TV":  {"type": ["object", "null"], "properties": {"insufficientie": {"type": ["string", "null"]}, "Ti_Vmax": {"type": ["string", "null"]}, "max_PG": {"type": ["string", "null"]}, "overige": {"type": "array", "items": {"type": "string"}}}, "required": ["insufficientie", "Ti_Vmax", "max_PG", "overige"], "additionalProperties": False},
                                "PV":  {"type": ["object", "null"], "properties": {"insufficientie": {"type": ["string", "null"]}, "overige": {"type": "array", "items": {"type": "string"}}}, "required": ["insufficientie", "overige"], "additionalProperties": False},
                            },
                            "required": ["AOV", "MV", "TV", "PV"],
                            "additionalProperties": False,
                        },
                        "aorta": {"type": ["object", "null"], "properties": {"AO_dimensies": {"type": ["string", "null"]}, "aorta_ascendens": {"type": ["string", "null"]}, "AO_root": {"type": ["string", "null"]}, "AO_boog": {"type": ["string", "null"]}, "AO_abdominalis": {"type": ["string", "null"]}}, "required": ["AO_dimensies", "aorta_ascendens", "AO_root", "AO_boog", "AO_abdominalis"], "additionalProperties": False},
                        "VCI": {"type": ["object", "null"], "properties": {"collaps": {"type": ["string", "null"]}, "geschatte_CVD": {"type": ["string", "null"]}, "sPAP": {"type": ["string", "null"]}, "overige": {"type": "array", "items": {"type": "string"}}}, "required": ["collaps", "geschatte_CVD", "sPAP", "overige"], "additionalProperties": False},
                        "PHT": {"type": ["object", "null"], "properties": {"sPAP": {"type": ["string", "null"]}, "secundaire_aanwijzingen": {"type": ["string", "null"]}}, "required": ["sPAP", "secundaire_aanwijzingen"], "additionalProperties": False},
                        "PE": {"type": ["string", "null"]},
                        "overige_bevindingen": {"type": "array", "items": {"type": "string"}},
                        "conclusie": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["datum", "LV", "RV", "LA", "RA", "kleppen", "aorta", "VCI", "PHT", "PE", "overige_bevindingen", "conclusie"],
                    "additionalProperties": False,
                },
                "TEE": {
                    "type": ["object", "null"],
                    "properties": {
                        "datum": {"type": ["string", "null"]},
                        "LV": {"type": ["object", "null"], "properties": {"dimensies": {"type": ["string", "null"]}, "LVIDd": {"type": ["string", "null"]}, "hypertrofie": {"type": ["string", "null"]}, "systolische_functie": {"type": ["string", "null"]}, "EF": {"type": ["string", "null"]}, "diastolische_functie": {"type": ["string", "null"]}, "E_e_prime": {"type": ["string", "null"]}, "E_A_verhouding": {"type": ["string", "null"]}, "S_D_verhouding": {"type": ["string", "null"]}, "RWBS": {"type": ["string", "null"]}, "overige": {"type": "array", "items": {"type": "string"}}}, "required": ["dimensies", "LVIDd", "hypertrofie", "systolische_functie", "EF", "diastolische_functie", "E_e_prime", "E_A_verhouding", "S_D_verhouding", "RWBS", "overige"], "additionalProperties": False},
                        "RV": {"type": ["object", "null"], "properties": {"dimensies": {"type": ["string", "null"]}, "functie": {"type": ["string", "null"]}, "TAPSE": {"type": ["string", "null"]}, "S_prime": {"type": ["string", "null"]}, "overige": {"type": "array", "items": {"type": "string"}}}, "required": ["dimensies", "functie", "TAPSE", "S_prime", "overige"], "additionalProperties": False},
                        "LA": {"type": ["object", "null"], "properties": {"dimensies": {"type": ["string", "null"]}, "LAVI": {"type": ["string", "null"]}, "LAESV_index": {"type": ["string", "null"]}, "volume": {"type": ["string", "null"]}, "overige": {"type": "array", "items": {"type": "string"}}}, "required": ["dimensies", "LAVI", "LAESV_index", "volume", "overige"], "additionalProperties": False},
                        "RA": {"type": ["object", "null"], "properties": {"dimensies": {"type": ["string", "null"]}, "RAVI": {"type": ["string", "null"]}, "RAESV": {"type": ["string", "null"]}, "area": {"type": ["string", "null"]}, "overige": {"type": "array", "items": {"type": "string"}}}, "required": ["dimensies", "RAVI", "RAESV", "area", "overige"], "additionalProperties": False},
                        "kleppen": {
                            "type": ["object", "null"],
                            "properties": {
                                "AOV": {"type": ["object", "null"], "properties": {"morfologie": {"type": ["string", "null"]}, "opening": {"type": ["string", "null"]}, "gradienten": {"type": ["string", "null"]}, "insufficientie": {"type": ["string", "null"]}, "AI_P1_2t": {"type": ["string", "null"]}, "AI_end_d_velocity": {"type": ["string", "null"]}, "overige": {"type": "array", "items": {"type": "string"}}}, "required": ["morfologie", "opening", "gradienten", "insufficientie", "AI_P1_2t", "AI_end_d_velocity", "overige"], "additionalProperties": False},
                                "MV":  {"type": ["object", "null"], "properties": {"morfologie": {"type": ["string", "null"]}, "mean_PG": {"type": ["string", "null"]}, "insufficientie": {"type": ["string", "null"]}, "calcificatie_annulus": {"type": ["string", "null"]}, "overige": {"type": "array", "items": {"type": "string"}}}, "required": ["morfologie", "mean_PG", "insufficientie", "calcificatie_annulus", "overige"], "additionalProperties": False},
                                "TV":  {"type": ["object", "null"], "properties": {"insufficientie": {"type": ["string", "null"]}, "Ti_Vmax": {"type": ["string", "null"]}, "max_PG": {"type": ["string", "null"]}, "overige": {"type": "array", "items": {"type": "string"}}}, "required": ["insufficientie", "Ti_Vmax", "max_PG", "overige"], "additionalProperties": False},
                                "PV":  {"type": ["object", "null"], "properties": {"insufficientie": {"type": ["string", "null"]}, "overige": {"type": "array", "items": {"type": "string"}}}, "required": ["insufficientie", "overige"], "additionalProperties": False},
                            },
                            "required": ["AOV", "MV", "TV", "PV"],
                            "additionalProperties": False,
                        },
                        "aorta": {"type": ["object", "null"], "properties": {"AO_dimensies": {"type": ["string", "null"]}, "aorta_ascendens": {"type": ["string", "null"]}, "AO_root": {"type": ["string", "null"]}, "AO_boog": {"type": ["string", "null"]}, "AO_abdominalis": {"type": ["string", "null"]}}, "required": ["AO_dimensies", "aorta_ascendens", "AO_root", "AO_boog", "AO_abdominalis"], "additionalProperties": False},
                        "VCI": {"type": ["object", "null"], "properties": {"collaps": {"type": ["string", "null"]}, "geschatte_CVD": {"type": ["string", "null"]}, "sPAP": {"type": ["string", "null"]}, "overige": {"type": "array", "items": {"type": "string"}}}, "required": ["collaps", "geschatte_CVD", "sPAP", "overige"], "additionalProperties": False},
                        "PHT": {"type": ["object", "null"], "properties": {"sPAP": {"type": ["string", "null"]}, "secundaire_aanwijzingen": {"type": ["string", "null"]}}, "required": ["sPAP", "secundaire_aanwijzingen"], "additionalProperties": False},
                        "PE": {"type": ["string", "null"]},
                        "overige_bevindingen": {"type": "array", "items": {"type": "string"}},
                        "conclusie": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["datum", "LV", "RV", "LA", "RA", "kleppen", "aorta", "VCI", "PHT", "PE", "overige_bevindingen", "conclusie"],
                    "additionalProperties": False,
                },
                "cag": {
                    "type": ["object", "null"],
                    "properties": {
                        "datum":       {"type": ["string", "null"]},
                        "dominantie":  {"type": ["string", "null"]},
                        "natieve_coronairen": {
                            "type": ["object", "null"],
                            "properties": {
                                "RCA": {"type": ["object", "null"], "properties": {"bevinding": {"type": ["string", "null"]}, "stenose_locatie": {"type": "array", "items": {"type": "string"}}, "stenose_graad": {"type": "array", "items": {"type": "string"}}, "overige": {"type": "array", "items": {"type": "string"}}}, "required": ["bevinding", "stenose_locatie", "stenose_graad", "overige"], "additionalProperties": False},
                                "LM":  {"type": ["object", "null"], "properties": {"bevinding": {"type": ["string", "null"]}, "stenose_locatie": {"type": "array", "items": {"type": "string"}}, "stenose_graad": {"type": "array", "items": {"type": "string"}}, "overige": {"type": "array", "items": {"type": "string"}}}, "required": ["bevinding", "stenose_locatie", "stenose_graad", "overige"], "additionalProperties": False},
                                "RDA": {"type": ["object", "null"], "properties": {"bevinding": {"type": ["string", "null"]}, "stenose_locatie": {"type": "array", "items": {"type": "string"}}, "stenose_graad": {"type": "array", "items": {"type": "string"}}, "overige": {"type": "array", "items": {"type": "string"}}}, "required": ["bevinding", "stenose_locatie", "stenose_graad", "overige"], "additionalProperties": False},
                                "RCX": {"type": ["object", "null"], "properties": {"bevinding": {"type": ["string", "null"]}, "stenose_locatie": {"type": "array", "items": {"type": "string"}}, "stenose_graad": {"type": "array", "items": {"type": "string"}}, "overige": {"type": "array", "items": {"type": "string"}}}, "required": ["bevinding", "stenose_locatie", "stenose_graad", "overige"], "additionalProperties": False},
                            },
                            "required": ["RCA", "LM", "RDA", "RCX"],
                            "additionalProperties": False,
                        },
                        "collateralen":        {"type": ["string", "null"]},
                        "hemostase":           {"type": ["string", "null"]},
                        "complicaties":        {"type": ["string", "null"]},
                        "conclusie":           {"type": "array", "items": {"type": "string"}},
                        "beleid":              {"type": "array", "items": {"type": "string"}},
                        "overige_bevindingen": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["datum", "dominantie", "natieve_coronairen", "collateralen", "hemostase", "complicaties", "conclusie", "beleid", "overige_bevindingen"],
                    "additionalProperties": False,
                },
                "ecg":            {"type": ["string", "null"]},
                "beeldvorming": {
                    "type": "object",
                    "properties": {
                        "pet_ct": {
                            "type": "object",
                            "properties": {
                                "datum":  {"type": ["string", "null"]},
                                "regels": {"type": "array", "items": {"type": "string"}},
                            },
                            "required": ["datum", "regels"],
                            "additionalProperties": False,
                        },
                        "ct_coronairen": {
                            "type": "object",
                            "properties": {
                                "datum":  {"type": ["string", "null"]},
                                "regels": {"type": "array", "items": {"type": "string"}},
                            },
                            "required": ["datum", "regels"],
                            "additionalProperties": False,
                        },
                        "ct_thorax": {
                            "type": "object",
                            "properties": {
                                "datum":  {"type": ["string", "null"]},
                                "regels": {"type": "array", "items": {"type": "string"}},
                            },
                            "required": ["datum", "regels"],
                            "additionalProperties": False,
                        },
                        "mri_hart": {
                            "type": "object",
                            "properties": {
                                "datum":  {"type": ["string", "null"]},
                                "regels": {"type": "array", "items": {"type": "string"}},
                            },
                            "required": ["datum", "regels"],
                            "additionalProperties": False,
                        },
                        "overige_beeldvorming": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "modaliteit": {"type": "string"},
                                    "datum":      {"type": ["string", "null"]},
                                    "regels":     {"type": "array", "items": {"type": "string"}},
                                },
                                "required": ["modaliteit", "datum", "regels"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": ["pet_ct", "ct_coronairen", "ct_thorax", "mri_hart", "overige_beeldvorming"],
                    "additionalProperties": False,
                },
                "overige_onderzoeken": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "naam":      {"type": "string"},
                            "datum":     {"type": ["string", "null"]},
                            "conclusie": {"type": ["string", "null"]},
                            "status":    {"type": ["string", "null"]},
                        },
                        "required": ["naam", "datum", "conclusie", "status"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["TTE", "TEE", "cag", "ecg", "beeldvorming", "overige_onderzoeken"],
            "additionalProperties": False,
        },
        "laboratorium": {
            "type": "object",
            "properties": {
                "datum":        {"type": ["string", "null"]},
                "hb":           {"type": ["string", "null"]},
                "ht":           {"type": ["string", "null"]},
                "leukocyten":   {"type": ["string", "null"]},
                "trombocyten":  {"type": ["string", "null"]},
                "natrium":      {"type": ["string", "null"]},
                "kalium":       {"type": ["string", "null"]},
                "ureum":        {"type": ["string", "null"]},
                "kreatinine":   {"type": ["string", "null"]},
                "egfr":         {"type": ["string", "null"]},
                "glucose":      {"type": ["string", "null"]},
                "crp":          {"type": ["string", "null"]},
                "hs_troponine": {"type": ["string", "null"]},
                "ck":           {"type": ["string", "null"]},
                "ld":           {"type": ["string", "null"]},
                "asat":         {"type": ["string", "null"]},
                "alat":         {"type": ["string", "null"]},
                "cholesterol":  {"type": ["string", "null"]},
                "hdl":          {"type": ["string", "null"]},
                "ldl":          {"type": ["string", "null"]},
                "triglyceriden":{"type": ["string", "null"]},
                "lpa":          {"type": ["string", "null"]},
                "tsh":          {"type": ["string", "null"]},
                "nt_pro_bnp":   {"type": ["string", "null"]},
                "calcium":      {"type": ["string", "null"]},
                "overige":      {"type": "array", "items": {"type": "string"}},
            },
            "required": [
                "datum", "hb", "ht", "leukocyten", "trombocyten",
                "natrium", "kalium", "ureum", "kreatinine", "egfr",
                "glucose", "crp", "hs_troponine", "ck", "ld",
                "asat", "alat", "cholesterol", "hdl", "ldl",
                "triglyceriden", "lpa", "tsh", "nt_pro_bnp", "calcium", "overige"
            ],
            "additionalProperties": False,
        },
        "extractie_metadata": {
            "type": "object",
            "properties": {
                "ontbrekende_velden":  {"type": "array", "items": {"type": "string"}},
                "opmerkingen":         {"type": ["string", "null"]},
                "niet_geextraheerd":   {"type": "array", "items": {"type": "string"}},
            },
            "required": ["ontbrekende_velden", "opmerkingen"],
            "additionalProperties": False,
        },
    },
    "required": ["patient", "document", "medisch", "anamnese", "diagnostiek", "laboratorium", "extractie_metadata"],
    "additionalProperties": False,
}


SYSTEM_PROMPT = """Je bent een medisch documentanalyse-assistent voor een Nederlands ziekenhuis.
Je taak is om gestructureerde patiëntinformatie te extraheren uit tekst van medische documenten.

Structuurregels (VERPLICHT):
- Je output bevat EXACT de volgende 5 top-level sleutels en geen andere: patient, document, medisch, diagnostiek, extractie_metadata.
- Elke top-level sleutel is ALTIJD aanwezig in je output, ook als alle waarden null of [] zijn.
- Voeg GEEN extra sleutels toe buiten het opgegeven schema. Gebruik EXACT de sleutelnamen uit het schema.

Inhoudsregels:
- Je antwoord is ALLEEN ÉÉN geldig JSON-object (UTF-8), zonder Markdown, zonder tekst eromheen.
- Alle strings tussen dubbele aanhalingstekens ". Gebruik geen apostroffen als quote voor keys.
- Gebruik null voor velden die je na grondig zoeken écht niet vindt — verzin nooit inhoud.
- Voor elk ingevuld tekstveld: kopieer de woorden spelling-onveranderd uit het document (verbatim), geen parafrase of vertaling.
- Geen komma na het laatste veld van een object of array (geen trailing comma).
- BSN alleen als exact 9 cijfers als string — anders null.
- Datums als DD-MM-YYYY of null.
- Bij 'Geen' of 'geen' bij allergieën: gebruik een lege array [].
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
    output_format: "str | dict" = "json",
) -> str:
    kwargs: dict[str, Any] = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "options": {
            "num_predict": OLLAMA_MAX_TOKENS,
            "temperature": 0.05,
            "top_p": 0.85,
        },
        "format": output_format,
    }

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
    raw = _chat_extract(client, messages=messages, output_format="json")
    return _parse_llm_json(raw)


MULTI_SOURCE_USER_INSTRUCTIONS = """
[Context: de tekst hieronder bevat meerdere achtereenvolgende PDF-bronnen van DEZELFDE patiënt.]
Voeg de inhoud samen tot één JSON volgens het schema:
- patient: één consistent profiel; bij duidelijk tegenstrijdige identiteitsgegevens: kies de meest onderbouwde waarde uit de tekst of zet op null en vermeld kort in extractie_metadata.opmerkingen.
- document (datum, type, ziekenhuis, afdeling): waar mogelijk de belangrijkste of meest recente klinische brief; anders null met toelichting in extractie_metadata.
- medisch: voeg diagnoses, voorgeschiedenis, medicatie en allergieën uit alle bronnen samen; geen duplicaten tenzij de bronteksten letterlijk verschillen (dan beide verbatim).
- diagnostiek: voeg onderzoeksbevindingen samen per type; bij meerdere echo's kies de meest recente of vermeld beide datums.
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

    prompt_schema = copy.deepcopy(EXTRACTION_SCHEMA)
    prompt_template = copy.deepcopy(EXTRACTION_TEMPLATE)
    if not EXTRACT_NIET_GEEXTRAHEERD:
        prompt_schema["extractie_metadata"].pop("niet_geextraheerd", None)
        prompt_template["extractie_metadata"].pop("niet_geextraheerd", None)

    user_prompt = f"""{intro}

Vul onderstaand JSON-template in met data uit het document.
Regels:
- Verander GEEN sleutelnamen en voeg GEEN nieuwe sleutels toe.
- Vervang null door de gevonden waarde (verbatim uit tekst), of laat null staan als niet gevonden.
- Vervang [] door gevulde arrays, of laat [] als er niets is.

Velduitleg — waar te zoeken per veld:
{json.dumps(prompt_schema, indent=2, ensure_ascii=False)}

In te vullen template:
{json.dumps(prompt_template, indent=2, ensure_ascii=False)}
{MULTI_SOURCE_USER_INSTRUCTIONS if multi else ""}
Documenttekst:

{full_text}"""

    client = ollama.Client(host=OLLAMA_BASE_URL)
    raw_output = ""



    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    try:
        try:
            raw_output = _chat_extract(client, messages=messages, output_format=EXTRACTION_JSON_SCHEMA)
            logger.info("Structured output (JSON Schema) gebruikt")
        except ollama.ResponseError as schema_err:
            if "cannot unmarshal object" in str(schema_err) or "400" in str(schema_err):
                logger.warning(
                    "Ollama server ondersteunt JSON Schema format niet (upgrade naar ≥ 0.5.0 "
                    "voor betere schema-afdwinging) — terugvallen op format=json"
                )
                raw_output = _chat_extract(client, messages=messages, output_format="json")
            else:
                raise

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
