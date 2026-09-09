web: streamlit run app/app.py --server.port $PORT --server.address 0.0.0.0
api: uvicorn service.main:app --host 0.0.0.0 --port $PORT
