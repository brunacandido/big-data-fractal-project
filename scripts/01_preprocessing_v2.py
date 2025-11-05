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
# testing
default_parq_files = [
    # from TRAIN
    "s3a://ubs-datasets/FRACTAL/data/train/TRAIN-0436_6399-002955257.parquet",
    # "s3a://ubs-datasets/FRACTAL/data/train/TRAIN-0436_6399-002955299.parquet",
    # "s3a://ubs-datasets/FRACTAL/data/train/TRAIN-0436_6399-002955400.parquet",
    # "s3a://ubs-datasets/FRACTAL/data/train/TRAIN-0436_6399-002955533.parquet",
    # "s3a://ubs-datasets/FRACTAL/data/train/TRAIN-0436_6400-002210500.parquet",
    # "s3a://ubs-datasets/FRACTAL/data/train/TRAIN-0436_6400-002210515.parquet",
    # "s3a://ubs-datasets/FRACTAL/data/train/TRAIN-0436_6400-002210520.parquet",
    # "s3a://ubs-datasets/FRACTAL/data/train/TRAIN-0436_6400-002210573.parquet",
    # "s3a://ubs-datasets/FRACTAL/data/train/TRAIN-0436_6400-002210596.parquet",
    # from TEST
    "s3a://ubs-datasets/FRACTAL/data/test/TEST-0436_6384-002325248.parquet",
    # "s3a://ubs-datasets/FRACTAL/data/test/TEST-0436_6384-002325312.parquet",
    # "s3a://ubs-datasets/FRACTAL/data/test/TEST-0436_6384-002325319.parquet",
    # "s3a://ubs-datasets/FRACTAL/data/test/TEST-0436_6384-002325374.parquet",
    # "s3a://ubs-datasets/FRACTAL/data/test/TEST-0436_6384-002325394.parquet",
    # "s3a://ubs-datasets/FRACTAL/data/test/TEST-0436_6384-002325525.parquet",
    # "s3a://ubs-datasets/FRACTAL/data/test/TEST-0436_6385-003103610.parquet",
    # "s3a://ubs-datasets/FRACTAL/data/test/TEST-0436_6385-003103656.parquet",
    # "s3a://ubs-datasets/FRACTAL/data/test/TEST-0436_6385-003103675.parquet",
    # "s3a://ubs-datasets/FRACTAL/data/test/TEST-0436_6385-003103798.parquet",
    # from VAL
    # "s3a://ubs-datasets/FRACTAL/data/val/VAL-0436_6406-003134108.parquet",
    # "s3a://ubs-datasets/FRACTAL/data/val/VAL-0436_6407-002561599.parquet",
    # "s3a://ubs-datasets/FRACTAL/data/val/VAL-0436_6408-002409795.parquet",
    # "s3a://ubs-datasets/FRACTAL/data/val/VAL-0436_6411-002310826.parquet",
    # "s3a://ubs-datasets/FRACTAL/data/val/VAL-0436_6411-002310834.parquet",
    # "s3a://ubs-datasets/FRACTAL/data/val/VAL-0436_6413-003295466.parquet",
    # "s3a://ubs-datasets/FRACTAL/data/val/VAL-0436_6415-002578342.parquet",
    # "s3a://ubs-datasets/FRACTAL/data/val/VAL-0436_6416-003028668.parquet",
    # "s3a://ubs-datasets/FRACTAL/data/val/VAL-0436_6417-002457489.parquet",
    "s3a://ubs-datasets/FRACTAL/data/val/VAL-0436_6420-002760111.parquet"
]


# default_parq_files = [
#     "s3a://ubs-datasets/FRACTAL/data/train/",
#     "s3a://ubs-datasets/FRACTAL/data/test/",
#     "s3a://ubs-datasets/FRACTAL/data/val/"
# ]

default_executor_mem = "8g"
default_driver_mem = "8g"
parq_cols = ["xyz", "Intensity", "Classification", "Red", "Green", "Blue", "Infrared"]

# -------------------------------------------------------
# Preprocessing Transformers
# -------------------------------------------------------
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
def main(args, test_mode=True):
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

    # Fit pipeline model
    pipeline_model = data_pipeline.fit(df_train)

    # Transform datasets
    df_train = pipeline_model.transform(df_train)
    df_test  = pipeline_model.transform(df_test)
    df_val   = pipeline_model.transform(df_val)

    # Select final columns
    final_cols = ["x", "y", "z_norm", "Intensity", "Classification",
                  "Red", "Green", "Blue", "Infrared", "NDVI"]
    df_train_final = df_train.select(*final_cols)
    df_test_final  = df_test.select(*final_cols)
    df_val_final   = df_val.select(*final_cols)

    print("\n=== Train Sample ===")
    df_train_final.show(5)
    print("\n=== Test Sample ===")
    df_test_final.show(5)
    print("\n=== Val Sample ===")
    df_val_final.show(5)
 
    taskmetrics.end()
    print("\n============< Transformation statistics >============")
    taskmetrics.print_report()

    spark.stop()

# -----------------------------------------------------
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
