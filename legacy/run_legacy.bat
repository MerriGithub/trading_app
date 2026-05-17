@echo off
cd /d "%~dp0"
streamlit run app_legacy.py --server.port 8501
