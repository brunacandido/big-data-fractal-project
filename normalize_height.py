from pyspark.sql import functions as F, Window
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


spark = SparkSession.builder 
# paste the rest here

df = spark.read.parquet("s3a://ubs-datasets/FRACTAL/data/train/TRAIN-0436_6399-002955257.parquet")

# getting all the coordinates in separate columns
df = df.withColumn("x", F.col("xyz")[0]) \
       .withColumn("y", F.col("xyz")[1]) \
       .withColumn("z", F.col("xyz")[2])

# normalizing height (z) by subtracting the minimum value
min_z = df.agg(F.min("z").alias("min_z")).collect()[0]["min_z"]
df = df.withColumn("z_norm", F.col("z") - F.lit(min_z))
df.show(5)