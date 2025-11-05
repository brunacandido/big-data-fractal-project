# BIG DATA 
# FRACTAL pipeline: remap classification, compute features, save selected columns
# Bruna Cândido ; Ethel Ogallo
# last update: 2025/11/05

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, count, lit, round as spark_round, min
from pyspark.ml import Transformer, Pipeline
from sparkmeasure import TaskMetrics
import argparse

# -------------------------------------------------------
# Default arguments
# -------------------------------------------------------
# list of the data folders in s3 bucket
default_parq_files = [
    "s3a://ubs-datasets/FRACTAL/data/train/",
    "s3a://ubs-datasets/FRACTAL/data/test/",
    "s3a://ubs-datasets/FRACTAL/data/val/"
]

default_executor_mem = "8g"
default_driver_mem = "8g"
parq_cols = ["xyz", "Intensity", "Classification", "Red", "Green", "Blue", "Infrared"]

# -------------------------------------------------------
# Preprocessing Transformers
# -------------------------------------------------------
# Transformer to remap the 'Classification' column
class remap_classification(Transformer):
    def _transform(self, df):
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

# Transformer to normalize height
class normalize_height(Transformer):
    """
    Normalize the height (z-coordinate) by subtracting the minimum z value.
    """
    def _transform(self, df):
        df = df.withColumn("x", col("xyz")[0]) \
               .withColumn("y", col("xyz")[1]) \
               .withColumn("z", col("xyz")[2])
        min_z = df.agg(min("z").alias("min_z")).collect()[0]["min_z"]
        df = df.withColumn("z_norm", col("z") - lit(min_z))
        return df

# Transformer to compute NDVI
class compute_ndvi(Transformer):
    """
    Compute the Normalized Difference Vegetation Index (NDVI) for the DataFrame.
    """
    def _transform(self, df):
        df = df.withColumn(
            "NDVI",
            when(
                (col("Infrared") + col("Red")) != 0,
                (col("Infrared") - col("Red")) / (col("Infrared") + col("Red"))
            ).otherwise(None)
        )
        return df

# Transformer to compute classification distribution
class compute_distribution(Transformer):
    """
    Compute the percentage distribution of 'Classification' column in the DataFrame.
    """
    def _transform(self, df):
        total_count = df.count()  # still triggers an action
        dist_df = df.groupBy("Classification") \
                    .agg((count("*") / lit(total_count) * 100).alias("percentage"))
        return dist_df

# ------------------------------------------------------
# Pipeline
# ------------------------------------------------------
data_pipeline = Pipeline(stages=[
    remap_classification(),
    normalize_height(),
    compute_ndvi()
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

    # ------------------------------
    # computing distribution of the land cover classification
    # ------------------------------
    # transformation is separate from the main pipeline because we wanted to just check classification distribution to
    # understand the distribution of the classification before and after remapping which would inform the fraction splitting
    # applied during the speed up phase.
    dist_train_df = compute_distribution().transform(df_train)
    dist_test_df  = compute_distribution().transform(df_test)
    dist_val_df   = compute_distribution().transform(df_val)

        # Combine distributions 
    dist_all = dist_train_df.join(dist_test_df, "Classification", "full_outer") \
                            .join(dist_val_df, "Classification", "full_outer") \
                            .fillna(0)

        # Add descriptions
    class_map = [
        (1, "Unclassified"),
        (2, "Ground"),
        (3, "Vegetation "),
        (4, "Building"),
        (5, "Water"),
        (6, "Bridge"),
        (7, "Permanent structures"),
        (8, "Filtered/Artifacts")]
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
    
    # print("\n============< Classification distribution >============")
    # dist_all.show(truncate=False)

    # --------------------------------------
    # Fit pipeline of transformations
    # --------------------------------------
    pipeline_model = data_pipeline.fit(df_train)

    # --------------------------------------
    # Transform all datasets
    # --------------------------------------
    df_train = pipeline_model.transform(df_train)
    df_test  = pipeline_model.transform(df_test)
    df_val   = pipeline_model.transform(df_val)

    # --------------------------------------
    # Select final features to save
    # --------------------------------------
    final_cols = ["x", "y", "z_norm", "Intensity", "Classification",
                  "Red", "Green", "Blue", "Infrared", "NDVI"]
    df_train_final = df_train.select(*final_cols)
    df_test_final  = df_test.select(*final_cols)
    df_val_final   = df_val.select(*final_cols)

    # --------------------------------------
    # Save processed datasets to s3 bucket ready for model training
    # --------------------------------------
    df_train_final.write.mode("overwrite").parquet("s3a://ubs-homes/erasmus/ethel/fractal/train")
    df_test_final.write.mode("overwrite").parquet("s3a://ubs-homes/erasmus/ethel/fractal/test")
    df_val_final.write.mode("overwrite").parquet("s3a://ubs-homes/erasmus/ethel/fractal/val")

    taskmetrics.end()
    print("\n============< Transformation statistics >============")
    taskmetrics.print_report()

    spark.stop()


# -------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FRACTAL Pipeline: preprocessing")
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
