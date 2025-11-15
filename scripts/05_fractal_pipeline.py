# ===========================================================
# BIG DATA - Scaling ML workflow with Spark using FRACTAL dataset
# Authors: Bruna Cândido ; Ethel Ogallo
# Last update: 15/11/2025
# ===========================================================

# -----------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------
# Spark SQL: manage distributed DataFrames

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, lit, min as spark_min

# Spark ML: machine learning library
from pyspark.ml import Pipeline, Transformer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# sparkmeasure: collects execution metrics (stages, tasks)
from sparkmeasure import TaskMetrics

# CLI argument parser
import argparse
import random
from datetime import datetime


# ----------------------------------------------------------------------
# Pre-processing Transformers
# ----------------------------------------------------------------------
class RemapClassification(Transformer):
    """
    Maps raw LiDAR classification codes into fewer aggregated classes.
    This helps simplify the prediction task by merging similar categories.
    """
    def _transform(self, df):
        return df.withColumn(
            "Classification",
            when(col("Classification") == 1, lit(1))
            .when(col("Classification") == 2, lit(2))
            .when(col("Classification").isin([3, 4, 5]), lit(3))
            .when(col("Classification") == 6, lit(4))
            .when(col("Classification") == 9, lit(5))
            .when(col("Classification") == 17, lit(6))
            .when(col("Classification") == 64, lit(7))
            .when(col("Classification").isin([65, 66]), lit(8))
            .otherwise(None)
        )


class NormalizeHeight(Transformer):
    """
    Extracts x, y, z coordinates from the 'xyz' array and normalizes the height
    (z value) by subtracting the minimum z in the dataset.
    This makes height information relative instead of absolute.
    """
    def _transform(self, df):
        # Extract XYZ components from array column
        df = df.withColumn("x", col("xyz")[0]) \
               .withColumn("y", col("xyz")[1]) \
               .withColumn("z", col("xyz")[2])
        # Compute global minimum height
        min_z = df.agg(spark_min("z").alias("min_z")).collect()[0]["min_z"]
        # Create normalized height column
        return df.withColumn("z_norm", col("z") - lit(min_z))

class ComputeNDVI(Transformer):
    """
    Computes the NDVI index using the standard formula:
    (NIR - Red) / (NIR + Red)
    Helps distinguish vegetation from non-vegetation.
    """
    def _transform(self, df):
        return df.withColumn(
            "NDVI",
            when((col("Infrared") + col("Red")) != 0,
                 (col("Infrared") - col("Red")) / (col("Infrared") + col("Red"))
            ).otherwise(None)
        )

# ----------------------------------------------------------------------
# File-level sampling
# ----------------------------------------------------------------------
def load_sample(spark, path, fraction, cols):
    """
    Load only a fraction of Parquet files (not rows) from a given directory.
    This avoids reading the entire dataset before sampling.
    """
    print(f"\n[INFO] Loading data from {path} with file fraction={fraction}")
    
    sc = spark.sparkContext
    hadoop_conf = sc._jsc.hadoopConfiguration()
    
    # Hadoop filesystem interface
    uri = sc._jvm.java.net.URI(path)
    fs = sc._jvm.org.apache.hadoop.fs.FileSystem.get(uri, hadoop_conf)
    file_path = sc._jvm.org.apache.hadoop.fs.Path(path)

    # List all parquet files
    all_files = [
        str(f.getPath()) for f in fs.listStatus(file_path)
        if str(f.getPath()).endswith(".parquet")
    ]
    if not all_files:
        raise ValueError(f"No parquet files found under {path}")

    # Choose sample of files
    num_files = max(1, int(len(all_files) * fraction))
    random.seed(42)
    selected_files = random.sample(all_files, num_files)

    # Read only the selected files
    print(f"[INFO] Loading {num_files}/{len(all_files)} files ({fraction*100:.1f}%)")
    return spark.read.parquet(*selected_files).select(*cols)


# ----------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------
def main(args):
    # Initialize Spark Session
    spark = (
        SparkSession.builder
        .appName(f'Fractal: frac={args.sample_fraction} exec={args.num_executors}')
        .config('spark.executor.instances', args.num_executors)
        .config('spark.executor.memory', args.executor_mem)
        .config('spark.driver.memory', args.driver_mem)
        .config('spark.executor.cores', args.executor_cores)
        .config('spark.dynamicAllocation.enabled', 'false')
        .config("spark.hadoop.fs.s3a.fast.upload", "true")
        .config("spark.hadoop.fs.s3a.multipart.size", "104857600")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.aws.credentials.provider",
                "com.amazonaws.auth.DefaultAWSCredentialsProviderChain")
        .config("spark.driver.maxResultSize", "512m")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.sql.shuffle.partitions", str(args.num_executors * args.executor_cores * 4))
        .config("spark.sql.files.maxPartitionBytes", "268435456")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.advisoryPartitionSizeInBytes", "134217728") 
    )
    spark = spark.getOrCreate()
    print(f"Spark session created with app name: {spark.sparkContext.appName}")
    spark.sparkContext.setLogLevel("WARN")

    # Start task-level metrics collection
    taskmetrics = TaskMetrics(spark)
    taskmetrics.begin()

    # input data path args
    if len(args.input) != 3:
        raise ValueError("--input must contain exactly 3 paths: train, val, test")
    train_path, val_path, test_path = args.input

    # ----------------------------------------------------------------------
    # Load datasets using file-level sampling
    # ----------------------------------------------------------------------
    parq_cols = ["xyz", "Intensity", "Classification", "Red", "Green", "Blue", "Infrared"]

    df_train = load_sample(spark, train_path, args.sample_fraction, parq_cols)
    df_val   = load_sample(spark, val_path,   args.sample_fraction, parq_cols)
    df_test  = load_sample(spark, test_path,  args.sample_fraction, parq_cols)

    # === REPARTITION ===
    repartitions = args.num_executors * args.executor_cores * 4
    df_train = df_train.repartition(repartitions)
    df_val   = df_val.repartition(repartitions)
    df_test  = df_test.repartition(repartitions)
    print(f"[INFO] Data repartitioned to {repartitions} partitions")

    # ----------------------------------------------------------------------
    # Preprocessing Pipeline
    # ----------------------------------------------------------------------
    assembler = VectorAssembler(
        inputCols=["x", "y", "z_norm", "Intensity", "Red", "Green", "Blue", "Infrared", "NDVI"],
        outputCol="features",
        handleInvalid="skip"
    )

    pipeline = Pipeline(stages=[
        RemapClassification(),
        NormalizeHeight(),
        ComputeNDVI(),
        assembler
    ])

    # Fit transformations on train set once
    print("Fitting preprocessing pipeline...")
    data_pipeline = pipeline.fit(df_train)

    # Apply transformations
    df_train = data_pipeline.transform(df_train)
    df_val   = data_pipeline.transform(df_val)
    df_test  = data_pipeline.transform(df_test)

    evaluator = MulticlassClassificationEvaluator(
        labelCol="Classification",
        predictionCol="prediction",
        metricName="accuracy"
    )

    # ----------------------------------------------------------------------
    # RandomForest hyperparameter tuning on val set
    # ----------------------------------------------------------------------
    best_acc = 0.0
    best_params = {}

    print("\n============< Validation Accuracy >============")
    for n in [10, 20, 30]:
        # Train RF model with current number of trees
        rf = RandomForestClassifier(
            labelCol="Classification", featuresCol="features", seed=42,
            numTrees=n, maxDepth=5
        )

        # Predict on validation set
        model = rf.fit(df_train)
        acc = evaluator.evaluate(model.transform(df_val))
        print(f"numTrees={n} → acc={acc:.4f}")

        # Track best performance
        if acc > best_acc:
            best_acc = acc
            best_params = {"numTrees": n}

    print(f"\nBest: numTrees={best_params['numTrees']} (acc={best_acc:.4f})")

    # ----------------------------------------------------------------------
    # Evaluate best model on test set
    # ----------------------------------------------------------------------
    final_model = RandomForestClassifier(
        labelCol="Classification",
        featuresCol="features",
        seed=42,
        **best_params,
        maxDepth=5
    ).fit(df_train)

    test_acc = evaluator.evaluate(final_model.transform(df_test))
    print(f"\nTEST ACCURACY: {test_acc:.4f}")

    taskmetrics.end()
    print("\n============< Task Metrics >============")
    taskmetrics.print_report()

    spark.stop()
    print("Job finished.")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FRACTAL ML Pipeline - Scaling Experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", nargs="+", required=True,
                        help="train_path val_path test_path")
    parser.add_argument("--executor-mem", default="8g", help="Fixed: 8g")
    parser.add_argument("--driver-mem", default="6g", help="Fixed: 6g")
    parser.add_argument("--executor-cores", type=int, default=2, help="Fixed: 6g")
    parser.add_argument("--num-executors", type=int, default=8, help="Fixed: 6g")
    parser.add_argument("--sample-fraction", type=float, default=0.01, help="Fixed: 6g")
    parser.add_argument("--enable-stage-metrics", action="store_true")
    args = parser.parse_args()
    main(args)


# ==========================================
# NOTES ON SPARK SUBMISSION COMMANDS
# ==========================================
# We used these configurations to perform scaling experiments:

# Number of executors: 8, 16, 24, 30 and 32
# Number of Fractions: 0.01, 0.03, 0.05
# Number or clusters nodes: 8  
# Number of cores per executor: 2   # constant across experiments
# Memory per executor: 8g  # constant across experiments
# Driver memory: 6g  # constant across experiments


# With this spark-submit command:

# spark-submit \
#   --master yarn \
#   --deploy-mode cluster \
#   --packages ch.cern.sparkmeasure:spark-measure_2.12:0.27 \
#   --num-executors 8 \
#   full_pipeline_v4.py \
#   --input s3a://ubs-datasets/FRACTAL/data/train/ s3a://ubs-datasets/FRACTAL/data/val/ s3a://ubs-datasets/FRACTAL/data/test/ \
#   --num-executors 8 \
#   --sample-fraction 0.01 \


