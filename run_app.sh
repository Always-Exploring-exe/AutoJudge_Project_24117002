#!/bin/bash

echo "========================================"
echo "  AutoJudge Streamlit App Launcher"
echo "========================================"
echo ""
echo "Starting Streamlit application..."
echo ""

cd "$(dirname "$0")"
streamlit run src/web/streamlit_app.py

