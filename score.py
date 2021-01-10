import json
import os

import joblib
import numpy as np
import scipy.sparse
import xgboost as xgb


def init():
    global model
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It's the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION).
    # For multiple models, it points to the folder containing all deployed models (./azureml-models).
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'xgboost_model.pkl')
    model = joblib.load(model_path)


def run(raw_data):
    data = np.array(json.loads(raw_data)['data'])
    csr_data = scipy.sparse.csr_matrix(data)
    test_data = xgb.DMatrix(csr_data)

    # Make prediction.
    y_hat = model.predict(test_data)
    # You can return any data type as long as it's JSON-serializable.
    return y_hat.tolist()
