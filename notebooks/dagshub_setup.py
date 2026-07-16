import mlflow
import dagshub

mlflow.set_tracking_uri('https://dagshub.com/balaram.sahu8/mlops_mini_project.mlflow')
dagshub.init(repo_owner='balaram.sahu8', repo_name='mlops_mini_project', mlflow=True)


with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)