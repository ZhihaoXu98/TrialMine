.PHONY: setup download index serve ui test lint

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

test:
	pytest tests/

lint:
	ruff check src/
