# Databricks notebook source
# install dependencies
%pip install -e ..
%pip install git+https://github.com/end-to-end-mlops-databricks-3/marvelous@0.1.0

# COMMAND ----------

#restart python
%restart_python

# COMMAND ----------

# system path update, must be after %restart_python
# caution! This is not a great approach
from pathlib import Path
import sys
sys.path.append(str(Path.cwd().parent / 'src'))

# COMMAND ----------

# A better approach (this file must be present in a notebook folder, achieved via synchronization)
%pip install hotel_reserves-1.0.1-py3-none-any.whl

# COMMAND ----------

from pyspark.sql import SparkSession
import mlflow

from hotel_reserves.config import ProjectConfig
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from lightgbm import LGBMRegressor, LGBMClassifier
from mlflow.models import infer_signature
from marvelous.common import is_databricks
from dotenv import load_dotenv
import os
from mlflow import MlflowClient
import pandas as pd
from hotel_reserves import __version__
from mlflow.utils.environment import _mlflow_conda_env
from databricks import feature_engineering
from databricks.feature_engineering import FeatureFunction, FeatureLookup
from pyspark.errors import AnalysisException
import numpy as np
from datetime import datetime
import boto3


# COMMAND ----------

if not is_databricks():
    load_dotenv()
    profile = os.environ["PROFILE"]
    mlflow.set_tracking_uri(f"databricks://{profile}")
    mlflow.set_registry_uri(f"databricks-uc://{profile}")


config = ProjectConfig.from_yaml(config_path="../project_config.yml", env="dev")

# COMMAND ----------

config

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()
fe = feature_engineering.FeatureEngineeringClient()

train_set = spark.table(f"{config.catalog_name}.{config.schema_name}.train_set")
test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set")

# COMMAND ----------

display(test_set.limit(5))

# COMMAND ----------

# create feature table with information about hotel reservations

feature_table_name = f"{config.catalog_name}.{config.schema_name}.hotel_reserves_features_demo"
lookup_features = ["no_of_adults", "no_of_children", "no_of_week_nights", "lead_time"]


# COMMAND ----------

# Option 1: feature engineering client
feature_table = fe.create_table(
   name=feature_table_name,
   primary_keys=["Booking_ID"],
   df=train_set[["Booking_ID"]+lookup_features],
   description="Hotel Reserves features table",
)

spark.sql(f"ALTER TABLE {feature_table_name} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")

fe.write_table(
   name=feature_table_name,
   df=test_set[["Booking_ID"]+lookup_features],
   mode="merge",
)

# COMMAND ----------

# create feature table with information about hotel reservations
# Option 2: SQL

spark.sql(f"""
          CREATE OR REPLACE TABLE {feature_table_name}
          (Id STRING NOT NULL, OverallQual INT, GrLivArea INT, GarageCars INT);
          """)
# primary key on Databricks is not enforced!
try:
    spark.sql(f"ALTER TABLE {feature_table_name} ADD CONSTRAINT house_pk_demo PRIMARY KEY(Id);")
except AnalysisException:
    pass
spark.sql(f"ALTER TABLE {feature_table_name} SET TBLPROPERTIES (delta.enableChangeDataFeed = true);")
spark.sql(f"""
          INSERT INTO {feature_table_name}
          SELECT Booking_ID, no_of_adults, no_of_children, no_of_week_nights, lead_time
          FROM {config.catalog_name}.{config.schema_name}.train_set
          """)
spark.sql(f"""
          INSERT INTO {feature_table_name}
          SELECT Booking_ID, no_of_adults, no_of_children, no_of_week_nights, lead_time
          FROM {config.catalog_name}.{config.schema_name}.test_set
          """)

# COMMAND ----------

# create feature function
# docs: https://docs.databricks.com/aws/en/sql/language-manual/sql-ref-syntax-ddl-create-sql-function

# problems with feature functions:
# functions are not versioned 
# functions may behave differently depending on the runtime (and version of packages and python)
# there is no way to enforce python version & package versions for the function 
# this is only supported from runtime 17
# advised to use only for simple calculations

function_name = f"{config.catalog_name}.{config.schema_name}.calculate_hotel_bookings_demo"
print(function_name)

# COMMAND ----------


# Option 1: with Python
# spark.sql(f"""
#         CREATE OR REPLACE FUNCTION {function_name}(year_built BIGINT)
#         RETURNS INT
#         LANGUAGE PYTHON AS
#         $$
#         from datetime import datetime
#         return datetime.now().year - year_built
#         $$
#         """)


spark.sql(f"""
        CREATE OR REPLACE FUNCTION {function_name}(no_of_adults INT, no_of_children INT)
        RETURNS INT
        LANGUAGE PYTHON AS
        $$
        return no_of_adults + no_of_children
        $$
        """)

# COMMAND ----------

# it is possible to define simple functions in sql only without python
# Option 2
# spark.sql(f"""
#         CREATE OR REPLACE FUNCTION {function_name}_sql (year_built INT)
#         RETURNS INT
#         RETURN year(current_date()) - year_built;
#         """)

# COMMAND ----------

# execute function
# spark.sql(f"SELECT {function_name}_sql(1960) as house_age;")

result = spark.sql(f"SELECT {function_name}(5,1) as total_guests;")
display(result)

# COMMAND ----------

# create a training set
# training_set = fe.create_training_set(
#     df=train_set.drop("OverallQual", "GrLivArea", "GarageCars"),
#     label=config.target,
#     feature_lookups=[
#         FeatureLookup(
#             table_name=feature_table_name,
#             feature_names=["OverallQual", "GrLivArea", "GarageCars"],
#             lookup_key="Id",
#                 ),
#         FeatureFunction(
#             udf_name=function_name,
#             output_name="house_age",
#             input_bindings={"year_built": "YearBuilt"},
#             ),
#     ],
#     exclude_columns=["update_timestamp_utc"],
#     )

training_set = fe.create_training_set(
    df=train_set.drop(*lookup_features),
    label=config.target,
    feature_lookups=[
        FeatureLookup(
            table_name=feature_table_name,
            feature_names=lookup_features,
            lookup_key="Booking_ID",
                ),
        FeatureFunction(
            udf_name=function_name,
            output_name="total_guests",
            input_bindings={"no_of_adults": "no_of_adults", "no_of_children": "no_of_children" },
            ),
    ],
    exclude_columns=["update_timestamp_utc"],
    )

# COMMAND ----------

training_set.load_df().display()

# COMMAND ----------

# Train & register a model
training_df = training_set.load_df().toPandas()
X_train = training_df[config.num_features + config.cat_features + ["total_guests"]]
y_train = training_df[config.target]

# COMMAND ----------

y_train.head()

# COMMAND ----------

pipeline = Pipeline(
        steps=[("preprocessor", ColumnTransformer(
            transformers=[("cat", OneHotEncoder(handle_unknown="ignore"),
                           config.cat_features)],
            remainder="passthrough")
            ),
               ("classifier", LGBMClassifier(**config.parameters))]
        )

pipeline.fit(X_train, y_train)

# COMMAND ----------

config.experiment_name_fe

# COMMAND ----------

# mlflow.set_experiment("/Shared/demo-model-fe")
mlflow.set_experiment(config.experiment_name_fe)
with mlflow.start_run(run_name="hotel-demo-run-model-fe",
                      tags={"git_sha": "1234567890abcd",
                            "branch": "week3"},
                            description="demo run for FE model logging") as run:
    # Log parameters and metrics
    run_id = run.info.run_id
    mlflow.log_param("model_type", "LightGBM with preprocessing")
    mlflow.log_params(config.parameters)

    # Log the model
    signature = infer_signature(model_input=X_train, model_output=pipeline.predict(X_train))
    fe.log_model(
                model=pipeline,
                flavor=mlflow.sklearn,
                artifact_path="lightgbm-pipeline-model-fe",
                training_set=training_set,
                signature=signature,
            )
    

# COMMAND ----------

model_name = f"{config.catalog_name}.{config.schema_name}.hotel_model_fe_demo"
model_version = mlflow.register_model(
    model_uri=f'runs:/{run_id}/lightgbm-pipeline-model-fe',
    name=model_name,
    tags={"git_sha": "1234567890abcd"})

# COMMAND ----------

features = [f for f in ["Booking_ID", "arrival_month_sin", "arrival_month_cos"] + config.num_features + config.cat_features if f not in lookup_features]
print(features)

# COMMAND ----------

# make predictions
features = [f for f in ["Booking_ID", "arrival_month_sin", "arrival_month_cos"] + config.num_features + config.cat_features if f not in lookup_features]
predictions = fe.score_batch(
    model_uri=f"models:/{model_name}/{model_version.version}",
    df=test_set[features]
)

# COMMAND ----------

predictions.select("prediction").show(5)

# COMMAND ----------

# from pyspark.sql.functions import col

# features = [f for f in ["Id"] + config.num_features + config.cat_features if f not in lookup_features]
# test_set_with_new_id = test_set.select(*features).withColumn(
#     "Booking_ID",
#     (col("Booking_ID").cast("long") + 1000000).cast("string")
# )

# predictions = fe.score_batch(
#     model_uri=f"models:/{model_name}/{model_version.version}",
#     df=test_set_with_new_id 
# )

# COMMAND ----------

# make predictions for a non-existing entry -> error!
predictions.select("prediction").show(5)

# COMMAND ----------

num_adults_function = f"{config.catalog_name}.{config.schema_name}.replace_no_of_adults_missing"
spark.sql(f"""
        CREATE OR REPLACE FUNCTION {num_adults_function}(no_of_adults INT)
        RETURNS INT
        LANGUAGE PYTHON AS
        $$
        if no_of_adults is None:
            return 0
        else:
            return no_of_adults
        $$
        """)

num_children_function = f"{config.catalog_name}.{config.schema_name}.replace_no_of_children_missing"
spark.sql(f"""
        CREATE OR REPLACE FUNCTION {num_children_function}(no_of_children INT)
        RETURNS INT
        LANGUAGE PYTHON AS
        $$
        if no_of_children is None:
            return 0
        else:
            return no_of_children
        $$
        """)

weeknights_function = f"{config.catalog_name}.{config.schema_name}.replace_no_of_weeknights_missing"
spark.sql(f"""
        CREATE OR REPLACE FUNCTION {weeknights_function}(no_of_week_nights INT)
        RETURNS INT
        LANGUAGE PYTHON AS
        $$
        if no_of_week_nights is None:
            return 2
        else:
            return no_of_week_nights
        $$
        """)

# COMMAND ----------

# what if we want to replace with a default value if entry is not found
# what if we want to look up value in another table? the logics get complex
# problems that arize: functions/ lookups always get executed (if statememt is not possible)
# it can get slow...

# step 1: create 3 feature functions

# step 2: redefine create training set

# try again

# create a training set
training_set = fe.create_training_set(
    df=train_set.drop("no_of_adults", "no_of_children", "no_of_week_nights"),
    label=config.target,
    feature_lookups=[
        FeatureLookup(
            table_name=feature_table_name,
            feature_names=["no_of_adults", "no_of_children", "no_of_week_nights"],
            lookup_key="Booking_ID",
            rename_outputs={"no_of_adults": "lookup_no_of_adults",
                            "no_of_children": "lookup_no_of_children",
                            "no_of_week_nights": "lookup_no_of_week_nights"}
                ),
        FeatureFunction(
            udf_name=num_adults_function,
            output_name="no_of_adults",
            input_bindings={"no_of_adults": "lookup_no_of_adults"},
            ),
        FeatureFunction(
            udf_name=num_children_function,
            output_name="no_of_children",
            input_bindings={"no_of_children": "lookup_no_of_children"},
        ),
        FeatureFunction(
            udf_name=weeknights_function,
            output_name="no_of_week_nights",
            input_bindings={"no_of_week_nights": "lookup_no_of_week_nights"},
        ),
        FeatureFunction(
            udf_name=function_name,
            output_name="total_people",
            input_bindings={"no_of_adults": "no_of_adults", "no_of_children": "no_of_children" },
            ),
    ],
    exclude_columns=["update_timestamp_utc"],
    )

# COMMAND ----------

# Train & register a model
training_df = training_set.load_df().toPandas()
X_train = training_df[config.num_features + config.cat_features + ["total_guests"]]
y_train = training_df[config.target]

#pipeline
pipeline = Pipeline(
        steps=[("preprocessor", ColumnTransformer(
            transformers=[("cat", OneHotEncoder(handle_unknown="ignore"),
                           config.cat_features)],
            remainder="passthrough")
            ),
               ("classifier", LGBMClassifier(**config.parameters))]
        )

pipeline.fit(X_train, y_train)

# COMMAND ----------

mlflow.set_experiment(config.experiment_name_fe)
with mlflow.start_run(run_name="demo-run-model-fe",
                      tags={"git_sha": "1234567890abcd",
                            "branch": "week3"},
                            description="demo run for FE model logging") as run:
    # Log parameters and metrics
    run_id = run.info.run_id
    mlflow.log_param("model_type", "LightGBM with preprocessing")
    mlflow.log_params(config.parameters)

    # Log the model
    signature = infer_signature(model_input=X_train, model_output=pipeline.predict(X_train))
    fe.log_model(
                model=pipeline,
                flavor=mlflow.sklearn,
                artifact_path="lightgbm-pipeline-model-fe",
                training_set=training_set,
                signature=signature,
            )
model_name = f"{config.catalog_name}.{config.schema_name}.model_fe_demo"
model_version = mlflow.register_model(
    model_uri=f'runs:/{run_id}/lightgbm-pipeline-model-fe',
    name=model_name,
    tags={"git_sha": "1234567890abcd"})

# COMMAND ----------

from pyspark.sql.functions import col

features = [f for f in ["Booking_ID", "arrival_month_sin", "arrival_month_cos"] + config.num_features + config.cat_features if f not in lookup_features]
# test_set_with_new_id = test_set.select(*features).withColumn(
#     "Id",
#     (col("Id").cast("long") + 1000000).cast("string")
# )

predictions = fe.score_batch(
    model_uri=f"models:/{model_name}/{model_version.version}",
    df=test_set 
)

# COMMAND ----------

# make predictions for a non-existing entry -> no error!
predictions.select("prediction").show(5)

# COMMAND ----------

import boto3

region_name = "eu-west-1"
aws_access_key_id = os.environ["aws_access_key_id"]
aws_secret_access_key = os.environ["aws_secret_access_key"]

client = boto3.client(
    'dynamodb',
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=region_name
)

# COMMAND ----------

response = client.create_table(
    TableName='HouseFeatures',
    KeySchema=[
        {
            'AttributeName': 'Id',
            'KeyType': 'HASH'  # Partition key
        }
    ],
    AttributeDefinitions=[
        {
            'AttributeName': 'Id',
            'AttributeType': 'S'  # String
        }
    ],
    ProvisionedThroughput={
        'ReadCapacityUnits': 5,
        'WriteCapacityUnits': 5
    }
)

print("Table creation initiated:", response['TableDescription']['TableName'])

# COMMAND ----------

# client.put_item(
#     TableName='HouseFeatures',
#     Item={
#         'Id': {'S': 'house_001'},
#         'OverallQual': {'N': '8'},
#         'GrLivArea': {'N': '2450'},
#         'GarageCars': {'N': '2'}
#     }
# )

# COMMAND ----------

# response = client.get_item(
#     TableName='HouseFeatures',
#     Key={
#         'Id': {'S': 'house_001'}
#     }
# )

# # Extract the item from the response
# item = response.get('Item')
# print(item)

# COMMAND ----------

# from itertools import islice

# rows = spark.table(feature_table_name).toPandas().to_dict(orient="records")

# def to_dynamodb_item(row):
#     return {
#         'PutRequest': {
#             'Item': {
#                 'Id': {'S': str(row['Id'])},
#                 'OverallQual': {'N': str(row['OverallQual'])},
#                 'GrLivArea': {'N': str(row['GrLivArea'])},
#                 'GarageCars': {'N': str(row['GarageCars'])}
#             }
#         }
#     }

# items = [to_dynamodb_item(row) for row in rows]

# def chunks(lst, n):
#     """Yield successive n-sized chunks from lst."""
#     for i in range(0, len(lst), n):
#         yield lst[i:i + n]

# for batch in chunks(items, 25):
#     response = client.batch_write_item(
#         RequestItems={
#             'HouseFeatures': batch
#         }
#     )
#     # Handle any unprocessed items if needed
#     unprocessed = response.get('UnprocessedItems', {})
#     if unprocessed:
#         print("Warning: Some items were not processed. Retry logic needed.")

# COMMAND ----------

# We ran into more limitations when we tried complex data types as output of a feature function
# and then tried to use it for serving
# al alternatve solution: using an external database (we use DynamoDB here)

# create a DynamoDB table
# insert records into dynamo DB & read from dynamoDB

# create a pyfunc model

# COMMAND ----------


# class HousePriceModelWrapper(mlflow.pyfunc.PythonModel):
#     """Wrapper class for machine learning models to be used with MLflow.

#     This class wraps a machine learning model for predicting house prices.
#     """

#     def __init__(self, model: object) -> None:
#         """Initialize the HousePriceModelWrapper.

#         :param model: The underlying machine learning model.
#         """
#         self.model = model

#     def predict(
#         self, context: mlflow.pyfunc.PythonModelContext, model_input: pd.DataFrame | np.ndarray
#     ) -> dict[str, float]:
#         """Make predictions using the wrapped model.

#         :param context: The MLflow context (unused in this implementation).
#         :param model_input: Input data for making predictions.
#         :return: A dictionary containing the adjusted prediction.
#         """
#         client = boto3.client('dynamodb',
#                                    aws_access_key_id=os.environ["aws_access_key_id"],
#                                    aws_secret_access_key=os.environ["aws_secret_access_key"],
#                                    region_name=os.environ["region_name"])
        
#         parsed = []
#         for lookup_id in model_input["Id"]:
#             raw_item = client.get_item(
#                 TableName='HouseFeatures',
#                 Key={'Id': {'S': lookup_id}})["Item"]     
#             parsed_dict = {key: int(value['N']) if 'N' in value else value['S']
#                       for key, value in raw_item.items()}
#             parsed.append(parsed_dict)
#         lookup_df=pd.DataFrame(parsed)
#         merged_df = model_input.merge(lookup_df, on="Id", how="left").drop("Id", axis=1)
        
#         merged_df["GarageCars"] = merged_df["GarageCars"].fillna(2)
#         merged_df["GrLivArea"] = merged_df["GrLivArea"].fillna(1000)
#         merged_df["OverallQual"] = merged_df["OverallQual"].fillna(5)
#         merged_df["house_age"] = datetime.now().year - merged_df["YearBuilt"]
#         predictions = self.model.predict(merged_df)

#         return [int(x) for x in predictions]

# COMMAND ----------

# custom_model = HousePriceModelWrapper(pipeline)

# COMMAND ----------

# features = [f for f in ["Id"] + config.num_features + config.cat_features if f not in lookup_features]
# data = test_set.select(*features).toPandas()
# data

# COMMAND ----------

# custom_model.predict(context=None, model_input=data)

# COMMAND ----------

# #log model
# mlflow.set_experiment("/Shared/demo-model-fe-pyfunc")
# with mlflow.start_run(run_name="demo-run-model-fe-pyfunc",
#                       tags={"git_sha": "1234567890abcd",
#                             "branch": "week2"},
#                             description="demo run for FE model logging") as run:
#     # Log parameters and metrics
#     run_id = run.info.run_id
#     mlflow.log_param("model_type", "LightGBM with preprocessing")
#     mlflow.log_params(config.parameters)

#     # Log the model
#     signature = infer_signature(model_input=data, model_output=custom_model.predict(context=None, model_input=data))
#     mlflow.pyfunc.log_model(
#                 python_model=custom_model,
#                 artifact_path="lightgbm-pipeline-model-fe",
#                 signature=signature,
#             )
    

# COMMAND ----------

# # predict
# mlflow.models.predict(f"runs:/{run_id}/lightgbm-pipeline-model-fe", data[0:1])