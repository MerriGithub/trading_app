#!/bin/bash
cd "$(dirname "$0")"
streamlit run app_legacy.py --server.port 8501
