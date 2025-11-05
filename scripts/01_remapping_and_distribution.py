# BIG DATA 
# FRACTAL pipeline: remap classification and compute distributions
# Bruna Cândido ; Ethel Ogallo
# last update: 2025/11/04

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, count, lit, round as spark_round, min
from pyspark.sql.window import Window
from sparkmeasure import TaskMetrics
import argparse

# -------------------------------------------------------
# Default arguments
# -------------------------------------------------------
#  # cluster files
default_parq_files = [
    "s3a://ubs-datasets/FRACTAL/data/train/",
    "s3a://ubs-datasets/FRACTAL/data/test/",
    "s3a://ubs-datasets/FRACTAL/data/val/"
]

default_executor_mem = "4g"
default_driver_mem = "4g"
parq_cols = ["xyz", "Intensity", "Classification", "Red", "Green", "Blue", "Infrared"]


# -------------------------------------------------------
# Preprocessing functions
# -------------------------------------------------------
# 1. Remapping function
def remap_classification(df):
    """
    Remap the 'Classification' column in the FRACTAL dataset.

    Mapping:
    1 -> 1 (Unclassified)
    2 -> 2 (Ground)
    3,4,5 -> 3 (Vegetation)
    6 -> 4 (Building)
    9 -> 5 (Water)
    17 -> 6 (Bridge)
    64 -> 7 (Permanent structures)
    65,66 -> 8 (Filtered/Artifacts)
    Any other value -> None
    """
    return df.withColumn(
        "Classification",
        when(col("Classification") == 1, lit(1))
        .when(col("Classification") == 2, lit(2))
        .when(col("Classification").isin(3,4,5), lit(3))
        .when(col("Classification") == 6, lit(4))
        .when(col("Classification") == 9, lit(5))
        .when(col("Classification") == 17, lit(6))
        .when(col("Classification") == 64, lit(7))
        .when(col("Classification").isin(65,66), lit(8))
        .otherwise(None)
    )

# 2. Compute percentage distribution
def compute_distribution(df, name):
    """
    Compute the percentage distribution of 'Classification' column in the DataFrame.

    Args:
        df (DataFrame): Input Spark DataFrame with 'Classification' column.
        name (str): Name of the column to store the percentages (e.g., "Train").

    Returns:
        DataFrame: A Spark DataFrame with columns:
            - Classification
            - <name> : percentage (not rounded, raw value)
    """
    total = df.count()
    return df.groupBy("Classification") \
             .agg((count("*") / lit(total) * 100).alias(name))


# Process dataset function
def process_dataset(spark, input_files, name):
    """
    apply all defined preprocessing functions

    Args:
        spark (SparkSession): Active Spark session.
        input_files (list): List of parquet files or folders to read.
        name (str): Name of the dataset (used for distribution column).

    Returns:
        tuple:
            - df (DataFrame): Transformed DataFrame (cached).
            - dist (DataFrame): Distribution DataFrame with percentages.
    """
    df = spark.read.parquet(*input_files).select(*parq_cols)
    df = remap_classification(df)
    dist = compute_distribution(df, name)
    return df, dist

# ------------------------------------------------------
# Main program
# ------------------------------------------------------
def main(args):
    # Spark session
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

    # Separate input files into datasets
    train_files = [f for f in args.input if "train" in f.lower()]
    test_files  = [f for f in args.input if "test" in f.lower()]
    val_files   = [f for f in args.input if "val" in f.lower()]
    
    taskmetrics.begin()
    
    # Process each dataset
    df_train, dist_train = process_dataset(spark, train_files, "train")
    df_test, dist_test   = process_dataset(spark, test_files, "test")
    df_val, dist_val     = process_dataset(spark, val_files, "val")

    # Combine distributions into one table
    dist_all = dist_train.join(dist_test, "Classification", "full_outer") \
        .join(dist_val, "Classification", "full_outer") \
        .fillna(0)  # missing values as 0%
    
    # Add human-readable descriptions
    class_map = [
        (1, "Unclassified"),
        (2, "Ground"),
        (3, "Vegetation "),
        (4, "Building"),
        (5, "Water"),
        (6, "Bridge"),
        (7, "Permanent structures"),
        (8, "Filtered/Artifacts")
    ]
    df_map = spark.createDataFrame(class_map, ["Classification", "Description"])

    dist_all = dist_all.join(df_map, "Classification", "left") \
                    .select(
                        "Classification",
                        "Description",
                        spark_round("train", 2).alias("Train"),
                        spark_round("test", 2).alias("Test"),
                        spark_round("val", 2).alias("Val")
                    ) \
                    .orderBy("Classification")
    
    taskmetrics.end()
    print("\n============< Transformation statistics >============")
    taskmetrics.print_report()
    print("\n============< Classification Distribution >============")
    dist_all.show(truncate=False)
    
    # Free memory
    df_train.unpersist()
    df_test.unpersist()
    df_val.unpersist()
    spark.stop()


# -----------------------------------------------------
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
