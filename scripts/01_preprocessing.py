# BIG DATA 
# Land cover classification of FRACTAL dataset
# Loading, remap classification, and distribution per dataset (train/test/val)

# Bruna Cândido ; Ethel Ogallo
# last update: 2025/11/04


from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, count, lit, round as spark_round
from sparkmeasure import TaskMetrics

import argparse
from functools import reduce

# Default arguments
# local testing files on S3
default_parq_files = [
    # example small files for local testing
    # first 9 from TRAIN
    "s3a://ubs-datasets/FRACTAL/data/train/TRAIN-0436_6399-002955257.parquet",
    "s3a://ubs-datasets/FRACTAL/data/train/TRAIN-0436_6399-002955299.parquet",
    "s3a://ubs-datasets/FRACTAL/data/train/TRAIN-0436_6399-002955400.parquet",
    "s3a://ubs-datasets/FRACTAL/data/train/TRAIN-0436_6399-002955533.parquet",
    "s3a://ubs-datasets/FRACTAL/data/train/TRAIN-0436_6400-002210500.parquet",
    "s3a://ubs-datasets/FRACTAL/data/train/TRAIN-0436_6400-002210515.parquet",
    "s3a://ubs-datasets/FRACTAL/data/train/TRAIN-0436_6400-002210520.parquet",
    "s3a://ubs-datasets/FRACTAL/data/train/TRAIN-0436_6400-002210573.parquet",
    "s3a://ubs-datasets/FRACTAL/data/train/TRAIN-0436_6400-002210596.parquet",
    # first 9 from TEST
    "s3a://ubs-datasets/FRACTAL/data/test/TEST-0436_6384-002325248.parquet",
    "s3a://ubs-datasets/FRACTAL/data/test/TEST-0436_6384-002325312.parquet",
    "s3a://ubs-datasets/FRACTAL/data/test/TEST-0436_6384-002325319.parquet",
    "s3a://ubs-datasets/FRACTAL/data/test/TEST-0436_6384-002325374.parquet",
    "s3a://ubs-datasets/FRACTAL/data/test/TEST-0436_6384-002325394.parquet",
    "s3a://ubs-datasets/FRACTAL/data/test/TEST-0436_6384-002325525.parquet",
    "s3a://ubs-datasets/FRACTAL/data/test/TEST-0436_6385-003103610.parquet",
    "s3a://ubs-datasets/FRACTAL/data/test/TEST-0436_6385-003103656.parquet",
    "s3a://ubs-datasets/FRACTAL/data/test/TEST-0436_6385-003103675.parquet",
    "s3a://ubs-datasets/FRACTAL/data/test/TEST-0436_6385-003103798.parquet",
    # first 2 from VAL
    "s3a://ubs-datasets/FRACTAL/data/val/VAL-0436_6406-003134108.parquet",
    "s3a://ubs-datasets/FRACTAL/data/val/VAL-0436_6407-002561599.parquet",
    "s3a://ubs-datasets/FRACTAL/data/val/VAL-0436_6408-002409795.parquet",
    "s3a://ubs-datasets/FRACTAL/data/val/VAL-0436_6411-002310826.parquet",
    "s3a://ubs-datasets/FRACTAL/data/val/VAL-0436_6411-002310834.parquet",
    "s3a://ubs-datasets/FRACTAL/data/val/VAL-0436_6413-003295466.parquet",
    "s3a://ubs-datasets/FRACTAL/data/val/VAL-0436_6415-002578342.parquet",
    "s3a://ubs-datasets/FRACTAL/data/val/VAL-0436_6416-003028668.parquet",
    "s3a://ubs-datasets/FRACTAL/data/val/VAL-0436_6417-002457489.parquet",
    "s3a://ubs-datasets/FRACTAL/data/val/VAL-0436_6420-002760111.parquet"
]
# cluster files would be like:
# default_parq_files = [
#     "s3a://ubs-datasets/FRACTAL/data/train/",
#     "s3a://ubs-datasets/FRACTAL/data/test/",
#     "s3a://ubs-datasets/FRACTAL/data/val/"
# ]  
# Memory settings
default_executor_mem = "4g"
default_driver_mem = "4g"


# -------------------------
# Preprocessing function
# -------------------------

# 1. Remapping classification feature
def remap_classification(df, classification_col="Classification"):
    """
    Remap the 'Classification' column in the FRACTAL dataset
    """
    return df.withColumn(
        classification_col,
        when(col(classification_col) == 1, lit(1))
        .when(col(classification_col) == 2, lit(2))
        .when(col(classification_col).isin(3, 4, 5), lit(3))
        .when(col(classification_col) == 6, lit(4))
        .when(col(classification_col) == 9, lit(5))
        .when(col(classification_col) == 17, lit(6))
        .when(col(classification_col) == 64, lit(7))
        .when(col(classification_col).isin(65, 66), lit(8))
        .otherwise(None)
    )

# -------------------------
# Main program
# -------------------------
def main(args):
    
    input_files = args.input
    executor_mem = args.executor_mem
    driver_mem = args.driver_mem
    
    print("\n==============< Program parameters >===============")
    print("- input files= {}".format(input_files))
    print("- executor memory= {}".format(executor_mem))
    print("- driver memory= {}".format(driver_mem))
    print("===================================================")
    
    # Create Spark session
    spark = (
        SparkSession.builder
        .appName("Read and Remap FRACTAL files")
        .config("spark.hadoop.fs.s3a.fast.upload", "true")
        .config("spark.hadoop.fs.s3a.multipart.size", "104857600")
        .config("spark.executor.memory", executor_mem)
        .config("spark.driver.memory", driver_mem)
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")
    
    taskmetrics = TaskMetrics(spark)
    
    # Columns to read
    parq_cols = ["xyz", "Intensity", "Classification", "Red", "Green", "Blue", "Infrared"]
    
    print("\n============< Loading and transforming data >============")
    taskmetrics.begin()
    

    # Split input files by dataset
    train_files = [f for f in input_files if "/train/" in f]
    test_files  = [f for f in input_files if "/test/" in f]
    val_files   = [f for f in input_files if "/val/" in f]

    def load_and_remap(files):
        if not files:
            return None
        df = spark.read.parquet(*files).select(*parq_cols)
        df_remap = remap_classification(df, "Classification")
        df_remap.cache()
        df_remap.count()  # trigger caching
        return df_remap

    df_train = load_and_remap(train_files)
    df_test  = load_and_remap(test_files)
    df_val   = load_and_remap(val_files)
    
    # Compute percentages per dataset
    def get_percentage(df, name):
        if df is None:
            return None
        total = df.count()
        return df.groupBy("Classification")\
                 .agg(spark_round((count("*") / lit(total) * 100), 2).alias(name))

    dist_train = get_percentage(df_train, "Train")
    dist_test  = get_percentage(df_test, "Test")
    dist_val   = get_percentage(df_val, "Val")
    
    # Compute overall counts
    df_all = df_train.unionByName(df_test).unionByName(df_val)
    dist_count = df_all.groupBy("Classification").agg(count("*").alias("Count"))
    

    # Merge everything
    dists = [dist_count, dist_train, dist_test, dist_val]
    distribution_final = reduce(lambda a, b: a.join(b, "Classification", "outer"), dists)
   
    # Add class descriptions
    class_map = [
        (1, "Unclassified"),
        (2, "Ground"),
        (3, "Vegetation"),
        (4, "Building"),
        (5, "Water"),
        (6, "Bridge"),
        (7, "Permanent structures"),
        (8, "Filtered/Artifacts")
    ]
    df_map = spark.createDataFrame(class_map, ["Classification", "Description"])

    distribution_final = (
        distribution_final
        .join(df_map, "Classification", "left")
        .select("Classification", "Description", "Count", "Train", "Test", "Val")
        .orderBy("Classification")
    )
    
    taskmetrics.end()
    print("\n============< Transformation statistics >============")
    taskmetrics.print_report()
    print("\n=====================================================")
    
    # Show final distribution
    print("\n============< Classification Distribution per Dataset >============")
    distribution_final.show(truncate=False)
    
    # Unpersist when done
    for df in [df_train, df_test, df_val]:
        if df is not None:
            df.unpersist()
    
    spark.stop()


if __name__ == "__main__":
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="PySpark FRACTAL remapping program")
    parser.add_argument("--input",
                        required=False,
                        nargs="+",  # allows multiple files or folders
                        help="input file(s) or folder(s)",
                        default=default_parq_files)
    parser.add_argument("--executor-mem",
                        required=False, 
                        help="executor memory",
                        default=default_executor_mem)
    parser.add_argument("--driver-mem",
                        required=False, 
                        help="driver memory",
                        default=default_driver_mem)
    
    args = parser.parse_args()
    main(args)
