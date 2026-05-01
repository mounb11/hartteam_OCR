@echo off
title Hartteam OCR
cd /d "%~dp0"
set PYTHONUTF8=1

if not exist ".venv\Scripts\python.exe" (
    echo Geen venv gevonden. Eerst uitvoeren in deze map:
    echo   python -m venv .venv
    echo   .venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

echo Hartteam OCR starten...
echo Browser opent op http://localhost:8501  ^(sluit dit venster om te stoppen^)
echo.
".venv\Scripts\python.exe" -m streamlit run ui_app.py --browser.gatherUsageStats false
pause
