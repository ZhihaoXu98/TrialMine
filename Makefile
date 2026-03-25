.PHONY: setup download index serve ui mlflow compare test lint

setup:
	pip install -e ".[dev]"

download:
	python scripts/download_data.py

index:
	python scripts/build_index.py

serve:
	uvicorn TrialMine.api.app:app --reload

ui:
	streamlit run src/TrialMine/ui/app.py

mlflow:
	mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001

compare:
	python scripts/compare_methods.py

training-data:
	python scripts/generate_training_data.py

test:
	pytest tests/

lint:
	ruff check src/
