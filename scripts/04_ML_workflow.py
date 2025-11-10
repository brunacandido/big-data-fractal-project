# BIG DATA 
# FRACTAL pipeline: ML workflow; model training, evaluation, saving
# Bruna Cândido ; Ethel Ogallo
# last update: 2025/11/06

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml import Transformer, Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from sparkmeasure import TaskMetrics
import argparse

# -------------------------------------------------------
# Default arguments
# -------------------------------------------------------
# list of the data folders in s3 bucket
default_parq_files = [
    "s3a://ubs-homes/erasmus/ethel/fractal/train/",
    "s3a://ubs-homes/erasmus/ethel/fractal/test/",
    "s3a://ubs-homes/erasmus/ethel/fractal/val/"
]

default_executor_mem = "8g"
default_driver_mem = "8g"


# -------------------------------------------------------
# ML workflow
# -------------------------------------------------------
# vectorize features
vecAssembler = VectorAssembler(
    inputCols=['x','y','z_norm','Intensity','Red','Green','Blue','Infrared','NDVI'],
    outputCol="features",
    handleInvalid="skip"
)

# target label indexer
labelIndexer = StringIndexer(
    inputCol="Classification",
    outputCol="label",
    handleInvalid="skip"
)

# Random Forest classifier
rf = RandomForestClassifier(
    labelCol="label",
    featuresCol="features",
    numTrees=100,
    maxDepth=15,
    seed=42
)

# ------------------------------------------------------
# Pipeline
# ------------------------------------------------------
model_pipeline = Pipeline(stages=[
    vecAssembler,
    labelIndexer,
    rf
])

# ------------------------------------------------------
# Main program
# ------------------------------------------------------
def main(args):
    spark = (
        SparkSession.builder
        .appName("FRACTAL Pipeline")
        .config("spark.hadoop.fs.s3a.fast.upload", "true")
        .config("spark.hadoop.fs.s3a.multipart.size", "104857600")
        .config("spark.executor.memory", args.executor_mem)
        .config("spark.driver.memory", args.driver_mem)
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    taskmetrics = TaskMetrics(spark)

    # Separate input files
    train_files = [f for f in args.input if "train" in f.lower()]
    test_files  = [f for f in args.input if "test" in f.lower()]
    val_files   = [f for f in args.input if "val" in f.lower()]

    taskmetrics.begin()

    # Read datasets
    df_train = spark.read.parquet(*train_files).select(*parq_cols)
    df_test  = spark.read.parquet(*test_files).select(*parq_cols)
    df_val   = spark.read.parquet(*val_files).select(*parq_cols)

    
    # --------------------------------------
    # Fit model pipeline 
    # --------------------------------------
    # pipeline_model = data_pipeline.fit(df_train)


    # --------------------------------------
    # Save model
    # --------------------------------------


    # --------------------------------------
    # Loop to select 1%, 5%, 10 %, 25%, 50%, 100% of the data respecting distribution of classes
    # Save the task metrics for each loop iteration if possible
    # --------------------------------------

    taskmetrics.end()
    print("\n============< Transformation statistics >============")
    taskmetrics.print_report()

    spark.stop()


# -------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FRACTAL Pipeline: model workflow")
    parser.add_argument("--input",
                        nargs="+",
                        default=default_parq_files,
                        help="Input files or folders (train/test/val)")
    parser.add_argument("--executor-mem",
                        default=default_executor_mem,
                        help="Executor memory for Spark")
    parser.add_argument("--driver-mem",
                        default=default_driver_mem,
                        help="Driver memory for Spark")
    args = parser.parse_args()
    main(args)
