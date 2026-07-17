# updated app.py

from flask import Flask, render_template,request
import mlflow
import pickle
import os
import pandas as pd

import numpy as np
import pandas as pd
import os
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import dagshub

def lemmatization(text):
    """Lemmatize the text."""
    lemmatizer = WordNetLemmatizer()
    text = text.split()
    text = [lemmatizer.lemmatize(word) for word in text]
    return " ".join(text)

def remove_stop_words(text):
    """Remove stop words from the text."""
    stop_words = set(stopwords.words("english"))
    text = [word for word in str(text).split() if word not in stop_words]
    return " ".join(text)

def removing_numbers(text):
    """Remove numbers from the text."""
    text = ''.join([char for char in text if not char.isdigit()])
    return text

def lower_case(text):
    """Convert text to lower case."""
    text = text.split()
    text = [word.lower() for word in text]
    return " ".join(text)

def removing_punctuations(text):
    """Remove punctuations from the text."""
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = text.replace('؛', "")
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def removing_urls(text):
    """Remove URLs from the text."""
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def remove_small_sentences(df):
    """Remove sentences with less than 3 words."""
    for i in range(len(df)):
        if len(df.text.iloc[i].split()) < 3:
            df.text.iloc[i] = np.nan

def normalize_text(text):
    text = lower_case(text)
    text = remove_stop_words(text)
    text = removing_numbers(text)
    text = removing_punctuations(text)
    text = removing_urls(text)
    text = lemmatization(text)

    return text


# # Set up DagsHub credentials for MLflow tracking
# dagshub_token = os.getenv("DAGSHUB_PAT")
# if not dagshub_token:
#     raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

# os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
# os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

# dagshub_url = "https://dagshub.com"
# repo_owner = "campusx-official"
# repo_name = "mlops-mini-project"

# # Set up MLflow tracking URI
# mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

mlflow.set_tracking_uri('https://dagshub.com/balaram.sahu8/mlops_mini_project.mlflow')
dagshub.init(repo_owner='balaram.sahu8', repo_name='mlops_mini_project', mlflow=True)

app = Flask(__name__)

# load model from model registry
def get_latest_model_version(model_name):
    client = mlflow.MlflowClient()
    try:
        # Get all versions of the model
        versions = client.search_model_versions(f"name='{model_name}'")
        if versions:
            # Return the latest version
            return max([v.version for v in versions])
    except Exception as e:
        print(f"Error fetching model versions: {e}")
    return None

model_name = "my_model"
model_version = get_latest_model_version(model_name)

if model_version is None:
    print(f"Warning: Model '{model_name}' not found in registry. Please train and register a model first.")
    model = None
else:
    model_uri = f'models:/{model_name}/{model_version}'
    try:
        model = mlflow.pyfunc.load_model(model_uri)
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None

vectorizer = pickle.load(open('models/vectorizer.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html',result=None)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('index.html', result="Error: Model not loaded. Please train and register a model first.")

    text = request.form['text']
    original_text = text
    
    # clean
    text = normalize_text(text)

    # bow
    features = vectorizer.transform([text])

    # Convert sparse matrix to DataFrame
    features_df = pd.DataFrame.sparse.from_spmatrix(features)
    features_df = pd.DataFrame(features.toarray(), columns=[str(i) for i in range(features.shape[1])])

    # prediction
    result = model.predict(features_df)[0]
    
    # Get prediction probabilities
    proba = model.predict_proba(features_df)[0]
    sad_proba = proba[0]
    happy_proba = proba[1]

    # show
    return render_template('index.html', result=result, sad_proba=f"{sad_proba:.2%}", happy_proba=f"{happy_proba:.2%}")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")