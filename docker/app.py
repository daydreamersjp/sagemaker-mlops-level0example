from io import BytesIO 
import os
import json
import boto3
import sagemaker
import pandas as pd
import numpy as np
import joblib
import lightgbm

def preprocess(path):
    df = pd.read_csv(path)
    # drop the "Phone" feature column
    df = df.drop(["Phone"], axis=1)

    # Change the data type of "Area Code"
    df["Area Code"] = df["Area Code"].astype(object)

    # Drop several other columns
    df = df.drop(["Day Charge", "Eve Charge", "Night Charge", "Intl Charge"], axis=1)

    # Convert categorical variables into dummy/indicator variables.
    model_data = pd.get_dummies(df)
    model_data.columns = [c.lower().replace(' ','_') for c in model_data.columns]

    # Create one binary classification target column
    # In case the original data does not have "Churn?" column, do try-except.
    try:
        X = model_data.drop(["churn?_false.", "churn?_true."], axis=1)
        y = model_data["churn?_true."]    
        model_data = pd.concat([X, y], axis=1,)
    except:
        X = model_data
        y = pd.Series(np.array([np.nan]*X.shape[0]))

    return X, y

        
def handler(event, context):
    sess = sagemaker.Session()
    # Load model dumped in S3
    with BytesIO() as f:
        boto3.resource('s3').meta.client.download_fileobj(Bucket=sess.default_bucket(), Key=os.path.join('model-store','level0example', 'clf.joblib'), Fileobj=f)
        f.seek(0)
        clf_loaded = joblib.load(f)

    # Get prediction on new data. If the dummy variables are not complete, add unassigne dummy variables used in the model.
    X_new, _ = preprocess('test.csv')
    for c in clf_loaded.feature_name_:
        if np.isin(X_new.columns,c).sum()==0:
            X_new[c] = 0
            
    return {
        'statusCode': 200,
        'body': json.dumps(list(clf_loaded.predict_proba(X_new)[:,1]))
    }