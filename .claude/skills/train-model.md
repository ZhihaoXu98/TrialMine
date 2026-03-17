# Model Training Skill

When training any ML model for TrialMine:
1. Config MUST be in configs/training/{model_name}.yaml — never hardcode hyperparameters
2. Log ALL experiments to MLflow with: params, metrics per step, final metrics, model artifact
3. Save model to models/{model_name}/v{version}/
4. Create model card at docs/model-cards/{model_name}.md
5. Evaluate on held-out data and save results to docs/evaluation/
6. Print a clean summary table at the end showing key metrics
