from pyspark.sql import SparkSession
from sparkmeasure import TaskMetrics
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorAssembler

import argparse

spark = SparkSession.builder \
    .appName("RandomForestExample") \
    .getOrCreate()

default_executor_mem = "4g"
default_driver_mem = "4g"
parq_cols = ["x", "y", "z_norm", "Intensity", "Classification",
                  "Red", "Green", "Blue", "Infrared", "NDVI"]

default_parq_files = [
    "s3a://ubs-homes/erasmus/ethel/fractal/train/",
    "s3a://ubs-homes/erasmus/ethel/fractal/test/",
    "s3a://ubs-homes/erasmus/ethel/fractal/val/",
]

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
    # val_files   = [f for f in args.input if "val" in f.lower()]

    taskmetrics.begin()

    # Read datasets
    df_train = spark.read.parquet(*train_files).select(*parq_cols)
    df_test  = spark.read.parquet(*test_files).select(*parq_cols)
    # df_val   = spark.read.parquet(*val_files).select(*parq_cols)
    # List all feature columns (excluding your target label)
    feature_cols_train = [col for col in df_train.columns if col != "Classification"]
    feature_cols_train = feature_cols_train.na.fill(0, subset=feature_cols_train)

    # for c in feature_cols_train:
    #     data = data.withColumn(c, when(col(c).isNull(), 0).otherwise(col(c)))
    #     data = data.withColumn(c, when(col(c).isNaN(), 0).otherwise(col(c)))

    # Create feature vector
    assembler_train = VectorAssembler(inputCols=feature_cols_train, outputCol="features", handleInvalid="skip")
    assembled_data_train = assembler_train.transform(df_train)
    # Keep only label and features
    final_data_train = assembled_data_train.select("features", "Classification")

    feature_cols_test = [col for col in df_test.columns if col != "Classification"]
    # Create feature vector
    assembler_test = VectorAssembler(inputCols=feature_cols_test, outputCol="features")
    assembled_data_test = assembler_test.transform(df_test)
    # Keep only label and features
    final_data_test = assembled_data_test.select("features","Classification")

    rf = RandomForestClassifier(labelCol="Classification", featuresCol="features", numTrees=100)
    model = rf.fit(final_data_train)
    predictions = model.transform(final_data_test)
    predictions.select("Classification", "prediction", "probability").show(5)

    # model_pipeline = Pipeline(stages=[
    #     assembler_train,
    #     rf
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FRACTAL Pipeline:preprocessing")
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