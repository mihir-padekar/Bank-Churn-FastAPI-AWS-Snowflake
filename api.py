from scikeras.wrappers import KerasClassifier
import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import snowflake.connector

# Load pipeline
with open("pickle_files/final_pipeline.pkl", "rb") as f:
    final_pipeline = pickle.load(f)

# FastAPI app
app = FastAPI(title="Bank Customer Churn Prediction API")

# Snowflake connection parameters
SNOWFLAKE_CONFIG = {
    "user": "MIHIRPADEKAR",
    "password": "Moneyplant@777",
    "account": "NLXEVDI-CY29358",
    "warehouse": "ML_WAREHOUSE",
    "database": "CHURN_PROJECT",
    "schema": "PUBLIC",
    "role": "ACCOUNTADMIN"
}
# Endpoint for health check
@app.get("/")
def home():
    return {"message": "Bank Churn Prediction API is running 🚀"}

# Input model for CustomerId only
class CustomerIdRequest(BaseModel):
    CustomerId: int

TRAINING_COLUMNS = [
    'RowNumber', 'CustomerId', 'Surname', 'CreditScore', 'Geography', 
    'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 
    'HasCrCard', 'IsActiveMember', 'EstimatedSalary'
]
# Helper: fetch features from Snowflake
def get_customer_features(customer_id):
    conn = snowflake.connector.connect(**SNOWFLAKE_CONFIG)
    
    # Select only features, NOT the target 'Exited'
    query = f"""
        SELECT *
        FROM CUSTOMER_FEATURES
        WHERE "CustomerId" = {customer_id}
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df


# Helper: log prediction to Snowflake
def log_prediction(customer_id, prediction):
    conn = snowflake.connector.connect(**SNOWFLAKE_CONFIG)
    cursor = conn.cursor()
    cursor.execute(f"""
        INSERT INTO CUSTOMER_PREDICTIONS (CUSTOMERID, PREDICTION)
        VALUES ({customer_id}, {prediction})
    """)
    conn.commit()
    conn.close()

# Prediction endpoint using CustomerId
@app.post("/predict")
def predict(customer: CustomerIdRequest):
    # Fetch features from Snowflake
    df = get_customer_features(customer.CustomerId)
    
    if df.empty:
        return {"error": "CustomerId not found in Snowflake."}
    
    # Predict churn
    prediction = final_pipeline.predict(df)[0]
    
    # Log prediction in Snowflake
    log_prediction(customer.CustomerId, int(prediction))
    
    return {
        "CustomerId": customer.CustomerId,
        "prediction": int(prediction)  # 0 = Not Churn, 1 = Churn
    }
