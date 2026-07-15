python -m src.preprocess
python -m src.model  
python -m src.fairness
python -m src.explainer
streamlit run app.py --server.port $PORT --server.address 0.0.0.0