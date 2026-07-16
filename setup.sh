#!/bin/bash
python -m src.preprocess
python -m src.model
python -m src.fairness
streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true --browser.gatherUsageStats false