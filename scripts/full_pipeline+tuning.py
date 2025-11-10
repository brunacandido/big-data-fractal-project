# BIG DATA 
# FRACTAL pipeline: remap classification, compute features, save selected columns
# Bruna Cândido ; Ethel Ogallo
# last update: 2025/11/10

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, lit, min, round as spark_round, count
from pyspark.ml import Transformer, Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from sparkmeasure import TaskMetrics
import argparse

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
        min_z = df.agg(min("z").alias("min_z")).collect()[0]["min_z"]
        return df.withColumn("z_norm", col("z") - lit(min_z))

class ComputeNDVI(Transformer):
    def _transform(self, df):
        return df.withColumn(
            "NDVI",
            when((col("Infrared") + col("Red")) != 0,
                 (col("Infrared") - col("Red")) / (col("Infrared") + col("Red"))
            ).otherwise(None)
        )

class ComputeDistribution(Transformer):
    """Compute percentage distribution of 'Classification' column."""
    def _transform(self, df):
        total = df.count()
        return df.groupBy("Classification") \
                 .agg((count("*") / lit(total) * 100).alias("Percentage"))
    
# ---------------------------
# Default arguments
# ---------------------------
default_parq_files = [
    "s3a://ubs-homes/erasmus/ethel/fractal/train/",
    "s3a://ubs-homes/erasmus/ethel/fractal/test/",
    "s3a://ubs-homes/erasmus/ethel/fractal/val/"
]

default_executor_mem = "4g"
default_driver_mem = "4g"
parq_cols = ["xyz", "Intensity", "Classification", "Red", "Green", "Blue", "Infrared"]

# ---------------------------
# Main
# ---------------------------
def main(args):
    spark = (
        SparkSession.builder
        .appName("FRACTAL ML Pipeline - Single Reusable Pipeline")
        .config("spark.hadoop.fs.s3a.fast.upload", "true")
        .config("spark.hadoop.fs.s3a.multipart.size", "104857600")
        .config("spark.executor.memory", args.executor_mem)
        .config("spark.driver.memory", args.driver_mem)
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    taskmetrics = TaskMetrics(spark)
    taskmetrics.begin()

    # ---------------------------
    # Load datasets and sample fraction
    # ---------------------------
    train_files = [f for f in args.input if "train" in f.lower()]
    val_files = [f for f in args.input if "val" in f.lower()]
    test_files = [f for f in args.input if "test" in f.lower()]

    df_train = spark.read.parquet(*train_files).select(*parq_cols)
    df_val   = spark.read.parquet(*val_files).select(*parq_cols)
    df_test  = spark.read.parquet(*test_files).select(*parq_cols)

    # sample fraction
    df_train = df_train.sample(False, args.sample_fraction, seed=42)
    df_val = df_val.sample(False, args.sample_fraction, seed=42)
    df_test = df_test.sample(False, args.sample_fraction, seed=42)

    # ---------------------------
    # land cover class distribution
    # ---------------------------
    # Uncomment the lines below to see the distribution of land cover classes
    # dist_train = ComputeDistribution().transform(df_train)
    # dist_val   = ComputeDistribution().transform(df_val)
    # dist_test  = ComputeDistribution().transform(df_test)
    #
    # dist_all = dist_train.join(dist_val, "Classification", "full_outer") \
    #                      .join(dist_test, "Classification", "full_outer") \
    #                      .fillna(0)
    # dist_all.show(truncate=False)


    # ---------------------------
    # Define pipeline 
    # ---------------------------
    assembler = VectorAssembler(
        inputCols=["x","y","z_norm","Intensity","Red","Green","Blue","Infrared","NDVI"],
        outputCol="features",
        handleInvalid="skip"
    )

    rf = RandomForestClassifier(labelCol="Classification", featuresCol="features", seed=42)
    
    pipeline = Pipeline(stages=[RemapClassification(), 
                                NormalizeHeight(), 
                                ComputeNDVI(), 
                                assembler, 
                                rf])
    
    # ---------------------------
    # Hyperparameter tuning using val set
    # ---------------------------
    num_trees = [50, 100, 150]
    best_acc = 0
    best_num_trees = None
    evaluator = MulticlassClassificationEvaluator(labelCol="Classification", 
                                                  predictionCol="prediction", 
                                                  metricName="accuracy")
    
    print("\n============< Validation Accuracy >============")
    for n in num_trees:
        rf.setParams(numTrees=n)          # adjust RF param
        model = pipeline.fit(df_train)    # fit on train only
        val_pred = model.transform(df_val)
        acc = evaluator.evaluate(val_pred)
        print(f"Validation accuracy for numTrees={n}: {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            best_num_trees = n

    print("\n============< Best parameters >============")
    print(f"\n Best numTrees = {best_num_trees} with val accuracy = {best_acc:.4f}")

    # ---------------------------
    # Refit final pipeline on train + val
    # ---------------------------
    rf.setParams(numTrees=best_num_trees)
    final_model = pipeline.fit(df_train)  # refit pipeline on train

    # ---------------------------
    # Predict on test set
    # ---------------------------
    test_pred = final_model.transform(df_test)

    evaluator_f1 = MulticlassClassificationEvaluator(labelCol="Classification", predictionCol="prediction", metricName="f1")
    evaluator_precision = MulticlassClassificationEvaluator(labelCol="Classification", predictionCol="prediction", metricName="weightedPrecision")
    evaluator_recall = MulticlassClassificationEvaluator(labelCol="Classification", predictionCol="prediction", metricName="weightedRecall")

    acc_test = evaluator.evaluate(test_pred)
    f1_test = evaluator_f1.evaluate(test_pred)
    precision_test = evaluator_precision.evaluate(test_pred)
    recall_test = evaluator_recall.evaluate(test_pred)

    print("\n============< Test Set Metrics >============")
    print(f"Accuracy : {acc_test:.4f}")
    print(f"F1 Score : {f1_test:.4f}")
    print(f"Precision: {precision_test:.4f}")
    print(f"Recall   : {recall_test:.4f}")

    taskmetrics.end()
    print("\n============< Transformation Statistics >============")
    taskmetrics.print_report()
    spark.stop()


# ---------------------------
# CLI Entrypoint
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FRACTAL ML Pipeline")
    parser.add_argument("--input", nargs="+", default=default_parq_files, help="Input parquet files")
    parser.add_argument("--executor-mem", default=default_executor_mem)
    parser.add_argument("--driver-mem", default=default_driver_mem)
    parser.add_argument("--sample-fraction", type=float, default=0.01,
                        help="Fraction of dataset to sample (0 < fraction <= 1.0)")
    args = parser.parse_args()
    main(args)

