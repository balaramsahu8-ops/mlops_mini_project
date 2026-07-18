import os
import mlflow

def promote_model():
    # Set up DagsHub credentials for MLflow tracking
    dagshub_token = os.getenv("DAGSHUB_PAT")
    if not dagshub_token:
        raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

    dagshub_url = "https://dagshub.com"
    repo_owner = "balaram.sahu8"
    repo_name = "mlops_mini_project"

    # Set up MLflow tracking URI
    mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

    client = mlflow.MlflowClient()

    model_name = "my_model"

    # Get the latest version in staging
    all_versions = client.search_model_versions(f"name='{model_name}'")
    staging_versions = [v for v in all_versions if v.current_stage == 'Staging']
    if not staging_versions:
        raise ValueError(f"No staging versions found for model '{model_name}'")
    latest_version_staging = max(int(v.version) for v in staging_versions)

    # Archive the current production model
    prod_versions = [v for v in all_versions if v.current_stage == 'Production']
    for version in prod_versions:
        client.transition_model_version_stage(
            name=model_name,
            version=version.version,
            stage="Archived"
        )

    # Promote the new model to production
    client.transition_model_version_stage(
        name=model_name,
        version=latest_version_staging,
        stage="Production"
    )
    print(f"Model version {latest_version_staging} promoted to Production")

if __name__ == "__main__":
    promote_model()