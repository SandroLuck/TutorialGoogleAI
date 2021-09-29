# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: Python [conda env:root] *
#     language: python
#     name: conda-root-py
# ---

# # BIQUERRY

# %%bigquery
SELECT * FROM `bigquery-public-data.austin_311.311_service_requests` LIMIT 1000

# # BQ TO PANDAS

# +
from google.cloud import bigquery

bqclient = bigquery.Client()

# Download query results.
query_string = """
SELECT * FROM `bigquery-public-data.austin_311.311_service_requests` LIMIT 1000
"""
df = (
    bqclient.query(query_string)
    .result()
    .to_dataframe(
        # Optionally, explicitly request to use the BigQuery Storage API. As of
        # google-cloud-bigquery version 1.26.0 and above, the BigQuery Storage
        # API is used by default.
        create_bqstorage_client=True,
    )
)
# -

df

# # BQ TO GCS

df["unique_key"]=42
df.to_csv("gs://sandro-test/test.csv")

# # GCS TO PANDAS

import pandas as pd
other_df=pd.read_csv("gs://sandro-test/test.csv")

other_df

# # Archive you notebooks on GITHUB

# !pip install jupytext --upgrade
# To convert your notebook into a python file
# !jupytext --set-formats ipynb,py ai_tutorial.ipynb
# These two notebooks are now synced, e.g. when the notebook changes the python will to
# After executing this command
# !jupytext --sync ai_tutorial.ipynb   


# # Some Plotly

# +
# !pip install plotly --upgrade
# !pip install gcsfs
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

df = px.data.tips()
X = df.total_bill.values.reshape(-1, 1)

model = LinearRegression()
model.fit(X, df.tip)

x_range = np.linspace(X.min(), X.max(), 100)
y_range = model.predict(x_range.reshape(-1, 1))

fig = px.scatter(df, x='total_bill', y='tip', opacity=0.65)
fig.add_traces(go.Scatter(x=x_range, y=y_range, name='Regression Fit'))
fig.show()
# -


