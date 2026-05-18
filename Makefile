.PHONY: setup download index serve ui mlflow compare training-data finetune finetune-cross-encoder eval-cross-encoder demo-reranker train-ranker evaluate test lint stack stack-down stack-rebuild

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

finetune:
	python scripts/finetune_embeddings.py

finetune-cross-encoder:
	python scripts/finetune_cross_encoder.py

eval-cross-encoder:
	python scripts/evaluate_cross_encoder.py

demo-reranker:
	python scripts/demo_reranker.py

train-ranker:
	OMP_NUM_THREADS=1 python scripts/train_ranker.py

evaluate:
	OMP_NUM_THREADS=1 python scripts/evaluate.py

test:
	pytest tests/

lint:
	ruff check src/

# Bring up the full Docker stack (api + ui + es + redis + prometheus + grafana).
# Always rebuilds api + ui images first so source-code changes land in the
# containers — the only volumes mounted at runtime are ./data and ./models,
# so src/ changes are otherwise frozen at image-build time.
stack:
	docker compose up -d --build

# Force-rebuild the api + ui images without using build cache (slow; use
# after dependency changes in pyproject.toml or Dockerfile edits).
stack-rebuild:
	docker compose build --no-cache api ui
	docker compose up -d

stack-down:
	docker compose down
