"""
pipeline/hix_client.py
----------------------
Stap 5: Verstuur de FHIR Bundle naar HiX (on-premise EPD van ChipSoft).

HiX ondersteunt HL7 FHIR R4 via een REST API.
We sturen een FHIR transaction Bundle — HiX verwerkt alle resources in één keer.

Authenticatie:
  HiX gebruikt Bearer token authenticatie. Het token wordt geconfigureerd in .env.
  In productie: koppel dit aan SMART on FHIR of de OAuth2 flow die jullie IT-afdeling
  heeft ingericht met ChipSoft.

Let op:
  De HiX FHIR URL en token zijn on-premise — configureer deze in .env, nooit hardcoden.
"""

import json
import httpx
from loguru import logger

from config import HIX_FHIR_BASE_URL, HIX_API_TOKEN, HIX_TIMEOUT_SECONDS


def send_to_hix(fhir_bundle: dict) -> dict:
    """
    Verstuur een FHIR R4 Bundle naar de HiX FHIR endpoint.

    Parameters:
        fhir_bundle: dict — output van fhir_mapper.map_to_fhir()

    Returns:
        dict met:
            "success": bool
            "status_code": int (HTTP statuscode van HiX)
            "response": dict (antwoord van HiX, bijv. FHIR Bundle response)
            "error": str | None
    """
    endpoint = f"{HIX_FHIR_BASE_URL}/"   # FHIR transaction endpoint

    headers = {
        "Content-Type": "application/fhir+json",   # Verplicht FHIR content type
        "Accept": "application/fhir+json",
    }

    # Voeg authenticatie toe als het token geconfigureerd is
    if HIX_API_TOKEN:
        headers["Authorization"] = f"Bearer {HIX_API_TOKEN}"
    else:
        logger.warning("Geen HIX_API_TOKEN geconfigureerd — verzoek wordt zonder auth verstuurd")

    logger.info(f"Versturen naar HiX: {endpoint}")

    try:
        with httpx.Client(timeout=HIX_TIMEOUT_SECONDS) as client:
            response = client.post(
                endpoint,
                content=json.dumps(fhir_bundle, ensure_ascii=False),
                headers=headers,
            )

        # FHIR servers geven 200 of 201 terug bij succes
        if response.status_code in (200, 201):
            logger.success(f"  HiX: succesvol verstuurd (HTTP {response.status_code})")
            return {
                "success": True,
                "status_code": response.status_code,
                "response": response.json(),
                "error": None,
            }
        else:
            error_msg = f"HTTP {response.status_code}: {response.text[:500]}"
            logger.error(f"  HiX antwoordde met fout: {error_msg}")
            return {
                "success": False,
                "status_code": response.status_code,
                "response": None,
                "error": error_msg,
            }

    except httpx.ConnectError:
        error_msg = f"Kan HiX niet bereiken op {HIX_FHIR_BASE_URL} — is de server bereikbaar?"
        logger.error(f"  {error_msg}")
        return {"success": False, "status_code": None, "response": None, "error": error_msg}

    except httpx.TimeoutException:
        error_msg = f"HiX reageerde niet binnen {HIX_TIMEOUT_SECONDS} seconden"
        logger.error(f"  {error_msg}")
        return {"success": False, "status_code": None, "response": None, "error": error_msg}

    except Exception as e:
        error_msg = f"Onverwachte fout bij versturen naar HiX: {str(e)}"
        logger.error(f"  {error_msg}")
        return {"success": False, "status_code": None, "response": None, "error": error_msg}


def check_hix_connection() -> bool:
    """
    Controleer of HiX bereikbaar is (bijv. bij opstarten van de pipeline).
    Doet een simpele GET op de FHIR metadata endpoint.
    """
    metadata_url = f"{HIX_FHIR_BASE_URL}/metadata"
    try:
        with httpx.Client(timeout=10) as client:
            response = client.get(metadata_url)
        if response.status_code == 200:
            logger.info(f"HiX bereikbaar op {HIX_FHIR_BASE_URL}")
            return True
        else:
            logger.warning(f"HiX metadata endpoint gaf HTTP {response.status_code}")
            return False
    except Exception as e:
        logger.warning(f"HiX niet bereikbaar: {e}")
        return False
