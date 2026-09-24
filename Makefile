train:
	python src/modeling.py

v5:
	python -m src.v5_decisioning

v6:
	python -m src.v6_bank_grade

v7:
	python -m src.v7_tnex_decisioning

gate-v7:
	python scripts/v7_gate.py

v8:
	python -m src.v8_digital_lending_os

gate-v8:
	python scripts/v8_gate.py

v9:
	python -m src.v9_production_intelligence

gate-v9:
	python scripts/v9_gate.py

test:
	python -m unittest discover -s tests -v

api:
	uvicorn api.main:app --reload

app:
	streamlit run app/streamlit_app.py
