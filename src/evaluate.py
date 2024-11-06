import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import yaml
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from mlflow.models import infer_signature
import os

from urllib.parse import urlparse

os.environ['MLFLOW_TRACKING_URI']="https://dagshub.com/zakkou/Machine_Learning_Pipeline.mlflow"
os.environ['MLFLOW_TRACKING_USERNAME']="zakkou"
os.environ['MLFLOW_TRACKING_PASSWORD']="08dd8ff6c6595e8cc1247d9ac2cd6163d596dd5b"

params=yaml.safe_load(open("params.yaml"))["train"]

def evaluate (data_path,model_path):
    data=pd.read_csv(data_path)
    X=data.drop(columns=["Outcome"])
    y=data['Outcome']

    mlflow.set_tracking_uri("https://dagshub.com/zakkou/Machine_Learning_Pipeline.mlflow")

    ## load the model from the disk
    model=pickle.load(open(model_path,'rb'))

    predictions=model.predict(X)
    accuracy=accuracy_score(y,predictions)
    ## log metrics to MLFLOW

    mlflow.log_metric("accuracy",accuracy)
    print("Model accuracy:{accuracy}")

if __name__=="__main__":
    evaluate(params["data"],params["model"])