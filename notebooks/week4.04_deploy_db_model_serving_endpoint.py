# Databricks notebook source
# MAGIC %pip install hotel_reserves-1.0.1-py3-none-any.whl

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import os
import time
from typing import Dict, List

import requests
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession

# from house_price.config import ProjectConfig
# from house_price.serving.model_serving import ModelServing
from hotel_reserves.config import ProjectConfig
from hotel_reserves.serving.model_serving import ModelServing

# spark session

spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)

# get environment variables
os.environ["DBR_TOKEN"] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
os.environ["DBR_HOST"] = spark.conf.get("spark.databricks.workspaceUrl")

# Load project config
config = ProjectConfig.from_yaml(config_path="../project_config.yml", env="dev")
catalog_name = config.catalog_name
schema_name = config.schema_name

# COMMAND ----------

"""Model serving module."""

import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput,
)

workspace = WorkspaceClient()
model_name=f"{catalog_name}.{schema_name}.hotel_reserves_model_custom"
endpoint_name="aleung-hotel-bookings-custom-model-serving-db"
entity_version = '1' # registered model version

served_entities = [
    ServedEntityInput(
        entity_name=model_name,
        scale_to_zero_enabled=True,
        workload_size="Small",
        entity_version=entity_version,
        environment_vars={
                    "aws_access_key_id": "{{secrets/mlops_course/aws_access_key_id}}",
                    "aws_secret_access_key": "{{secrets/mlops_course/aws_secret_access_key}}",
                    "region_name": "eu-west-1",
                    }
    )
]

workspace.serving_endpoints.create(
        name=endpoint_name,
        config=EndpointCoreConfigInput(
            served_entities=served_entities,
        ),
    )

# COMMAND ----------

# Create a sample request body
required_columns = [
    "Booking_ID",
  "no_of_adults",
  "no_of_children",
  "no_of_weekend_nights",
  "no_of_week_nights",
  "required_car_parking_space",
  "lead_time",
  "arrival_year",
  "arrival_month",
  "arrival_month_sin",
  "arrival_month_cos",
  "arrival_date",
  "repeated_guest",
  "no_of_previous_cancellations",
  "no_of_previous_bookings_not_canceled",
  "avg_price_per_room",
  "no_of_special_requests",
  "type_of_meal_plan",
  "room_type_reserved",
  "market_segment_type"
]

spark = SparkSession.builder.getOrCreate()

train_set = spark.table(f"{catalog_name}.{schema_name}.train_set").toPandas()
sampled_records = train_set[required_columns].sample(n=1000, replace=True).to_dict(orient="records")
dataframe_records = [[record] for record in sampled_records]

print(train_set.dtypes)
print(dataframe_records[0])

# COMMAND ----------

# Call the endpoint with one sample record

"""
Each dataframe record in the request body should be list of json with columns looking like:

[{'LotFrontage': 78.0,
  'LotArea': 9317,
  'OverallQual': 6,
  'OverallCond': 5,
  'YearBuilt': 2006,
  'Exterior1st': 'VinylSd',
  'Exterior2nd': 'VinylSd',
  'MasVnrType': 'None',
  'Foundation': 'PConc',
  'Heating': 'GasA',
  'CentralAir': 'Y',
  'SaleType': 'WD',
  'SaleCondition': 'Normal'}]
"""

def call_endpoint(record):
    """
    Calls the model serving endpoint with a given input record.
    """
    serving_endpoint = f"https://{os.environ['DBR_HOST']}/serving-endpoints/aleung-hotel-bookings-custom-model-serving-db/invocations"

    response = requests.post(
        serving_endpoint,
        headers={"Authorization": f"Bearer {os.environ['DBR_TOKEN']}"},
        json={"dataframe_records": record},
    )
    return response.status_code, response.text


status_code, response_text = call_endpoint(dataframe_records[0])
print(f"Response Status: {status_code}")
print(f"Response Text: {response_text}")

# COMMAND ----------

# Load test
for i in range(len(dataframe_records)):
    status_code, response_text = call_endpoint(dataframe_records[i])
    print(f"Response Status: {status_code}")
    print(f"Response Text: {response_text}")
    time.sleep(0.2)