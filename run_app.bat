@echo off
echo ========================================
echo   AutoJudge Streamlit App Launcher
echo ========================================
echo.
echo Starting Streamlit application...
echo.

cd /d "%~dp0"
streamlit run src/web/streamlit_app.py

pause

