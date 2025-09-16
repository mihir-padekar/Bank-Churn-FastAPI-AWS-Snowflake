import pandas as pd
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas

# Load CSV
df = pd.read_csv("data/dataset.csv")

# Drop target column if present
if 'Exited' in df.columns:
    df.drop(columns=['Exited'], inplace=True)


# Connect to Snowflake
conn = snowflake.connector.connect(
    user='MIHIRPADEKAR',
    password='Moneyplant@777',
    account='NLXEVDI-CY29358',
    warehouse='ML_WAREHOUSE',
    database='CHURN_PROJECT',
    schema='PUBLIC',
    role='ACCOUNTADMIN'
)

# Upload DataFrame to Snowflake
success, nchunks, nrows, _ = write_pandas(conn, df, 'CUSTOMER_FEATURES')

print(f"Success: {success}, Rows inserted: {nrows}")

# Close connection
conn.close()
