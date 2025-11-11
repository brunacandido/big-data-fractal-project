# ===========================================================
# BIG DATA - Scaling ML workflow with Spark using FRACTAL dataset
# Authors: Bruna Cândido ; Ethel Ogallo
# Last update: 2025/11/11
# ===========================================================


from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, lit, min as spark_min, count
from pyspark.ml import Transformer, Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from sparkmeasure import TaskMetrics
import argparse
import random

# ---------------------------
# Preprocessing Transformers
# ---------------------------
class RemapClassification(Transformer):
    def _transform(self, df):
        return df.withColumn(
            "Classification",
            when(col("Classification") == 1, lit(1))
            .when(col("Classification") == 2, lit(2))
            .when(col("Classification").isin(3, 4, 5), lit(3))
            .when(col("Classification") == 6, lit(4))
            .when(col("Classification") == 9, lit(5))
            .when(col("Classification") == 17, lit(6))
            .when(col("Classification") == 64, lit(7))
            .when(col("Classification").isin(65, 66), lit(8))
            .otherwise(None)
        )

class NormalizeHeight(Transformer):
    def _transform(self, df):
        df = df.withColumn("x", col("xyz")[0]) \
               .withColumn("y", col("xyz")[1]) \
               .withColumn("z", col("xyz")[2])
        min_z = df.agg(spark_min("z").alias("min_z")).collect()[0]["min_z"]
        return df.withColumn("z_norm", col("z") - lit(min_z))

class ComputeNDVI(Transformer):
    def _transform(self, df):
        return df.withColumn(
            "NDVI",
            when((col("Infrared") + col("Red")) != 0,
                 (col("Infrared") - col("Red")) / (col("Infrared") + col("Red"))
            ).otherwise(None)
        )

# ---------------------------
# Default arguments
# ---------------------------
default_parq_files = [
    "s3a://ubs-datasets/FRACTAL/data/train/",
    "s3a://ubs-datasets/FRACTAL/data/test/",
    "s3a://ubs-datasets/FRACTAL/data/val/"
]

default_executor_mem = "4g"
default_driver_mem = "4g"
default_executor_cores = "2"

parq_cols = ["xyz", "Intensity", "Classification", "Red", "Green", "Blue", "Infrared"]

# ---------------------------
# File-level sampling of the dataset
# ---------------------------
def load_sample(spark, path, fraction, cols):
    """
    Load only a fraction of Parquet files (not rows) from a given directory.
    This avoids reading the entire dataset before sampling.
    """
    print(f"\n[INFO] Loading data from {path} with file fraction={fraction}")
    
    sc = spark.sparkContext
    hadoop_conf = sc._jsc.hadoopConfiguration()
    
    uri = sc._jvm.java.net.URI(path)
    fs = sc._jvm.org.apache.hadoop.fs.FileSystem.get(uri, hadoop_conf)
    file_path = sc._jvm.org.apache.hadoop.fs.Path(path)
    
    all_files = [
        str(f.getPath()) for f in fs.listStatus(file_path)
        if str(f.getPath()).endswith(".parquet")
    ]
    
    if not all_files:
        raise ValueError(f"No parquet files found under {path}")
    
    num_files = max(1, int(len(all_files) * fraction))
    random.seed(42)
    selected_files = random.sample(all_files, num_files)
    
    print(f"[INFO] Loading {num_files}/{len(all_files)} files ({fraction*100:.1f}%)")
    df = spark.read.parquet(*selected_files).select(*parq_cols)
    
    # Optional: check row count (expensive on huge data)
    # row_count = df.count()
    # print(f"[INFO] Loaded {row_count} rows from {path}")
    
    return df

# ---------------------------
# Main
# ---------------------------
def main(args):
    # Initialize Spark Session
    spark = (
        SparkSession.builder
        .appName(f'Fractal: {args.sample_fraction} and n_executor: {args.num_executors}')
        .config('spark.executor.instances', args.num_executors)
        .config('spark.executor.memory', args.executor_mem)
        .config('spark.driver.memory', args.driver_mem)
        .config('spark.executor.cores', args.executor_cores)
        .config('spark.dynamicAllocation.enabled', 'false')
        .config("spark.hadoop.fs.s3a.fast.upload", "true")
        .config("spark.hadoop.fs.s3a.multipart.size", "104857600")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    taskmetrics = TaskMetrics(spark)
    taskmetrics.begin()

    # ---------------------------
    # Load datasets using file-level sampling
    # ---------------------------
    train_files = [f for f in args.input if "train" in f.lower()]
    val_files   = [f for f in args.input if "val" in f.lower()]
    test_files  = [f for f in args.input if "test" in f.lower()]

    # load, sample, and repartition datasets
    df_train = load_sample(spark, train_files[0], args.sample_fraction, parq_cols)
    df_train = df_train.repartition(args.num_executors * int(args.executor_cores))

    df_val = load_sample(spark, val_files[0], args.sample_fraction, parq_cols)
    df_val = df_val.repartition(args.num_executors * int(args.executor_cores))

    df_test = load_sample(spark, test_files[0], args.sample_fraction, parq_cols)
    df_test = df_test.repartition(args.num_executors * int(args.executor_cores))

    # ---------------------------
    # Preprocessing pipeline
    # ---------------------------
    assembler = VectorAssembler(
        inputCols=["x","y","z_norm","Intensity","Red","Green","Blue","Infrared","NDVI"],
        outputCol="features",
        handleInvalid="skip"
    )

    data_pipeline = Pipeline(stages=[
        RemapClassification(),
        NormalizeHeight(),
        ComputeNDVI(),
        assembler
    ])

    fit_pipeline = data_pipeline.fit(df_train)
    df_train = fit_pipeline.transform(df_train)
    df_val   = fit_pipeline.transform(df_val)
    df_test  = fit_pipeline.transform(df_test)

    # ---------------------------
    # RandomForest hyperparameter tuning on val set
    # ---------------------------
    num_trees   = [10, 20, 30]
    best_acc    = 0
    best_params = {}
    evaluator = MulticlassClassificationEvaluator(
        labelCol="Classification",
        predictionCol="prediction",
        metricName="accuracy"
    )

    print("\n============< Validation Accuracy >============")
    for n in num_trees:
        rf = RandomForestClassifier(
            labelCol="Classification",
            featuresCol="features",
            seed=42,
            numTrees=n,
            maxDepth=5
        )
        model = rf.fit(df_train)
        val_pred = model.transform(df_val)
        acc = evaluator.evaluate(val_pred)
        print(f"Validation accuracy for numTrees={n}: {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            best_params = {"numTrees": n}

    print("\n============< Best parameters >============")
    print(f"Best params: numTrees={best_params['numTrees']} with val accuracy = {best_acc:.4f}")

    # ---------------------------
    # Refit final RF model on train set
    # ---------------------------
    opt_rf = RandomForestClassifier(
        labelCol="Classification",
        featuresCol="features",
        seed=42,
        **best_params,
        maxDepth=5
    )
    final_model = opt_rf.fit(df_train)

    # ---------------------------
    # Predict on test set
    # ---------------------------
    test_pred = final_model.transform(df_test)
    acc_test = evaluator.evaluate(test_pred)

    print("\n============< Test Set Metrics >============")
    print(f"Accuracy : {acc_test:.4f}")

    # ---------------------------
    # Task metrics
    # ---------------------------
    taskmetrics.end()
    print("\n============< Transformation Statistics >============")
    taskmetrics.print_report()
    spark.stop()


# ---------------------------
# CLI Entrypoint
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FRACTAL ML Pipeline", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", nargs="+", default=default_parq_files, help="Input parquet files")
    parser.add_argument("--executor-mem", default=default_executor_mem, help="Executor memory (e.g., '4g')")
    parser.add_argument("--driver-mem", default=default_driver_mem, help="Driver memory (e.g., '4g')")
    parser.add_argument("--executor-cores", default=default_executor_cores, help="Number of cores per executor")
    parser.add_argument("--sample-fraction", type=float, default=0.01, help="Fraction of dataset to sample (by files)")
    parser.add_argument("--num-executors", type=int, default=4, help="Number of Spark executors to use")
    args = parser.parse_args()
    main(args)



# num of executors  16 , 32, 64   with 16 nodes of cluster
# num of executors  8 , 16, 32   with 8 nodes of cluster
# num of cores  6, 3, 1
# memory per executor  20g, 14g, 7g

# spark-submit \
#   --master yarn --deploy-mode cluster \
#   --packages ch.cern.sparkmeasure:spark-measure_2.12:0.27 \
#   --num-executors 8 \
#   full_pipeline_v3.py \
#     --input s3a://ubs-datasets/FRACTAL/data/train/ s3a://ubs-datasets/FRACTAL/data/val/ s3a://ubs-datasets/FRACTAL/data/test/ \
#     --executor-mem 20g --driver-mem 20g --executor-cores 5 \
#     --num-executors 8 --sample-fraction 0.01



