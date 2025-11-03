from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("Fractal") \
    .config("spark.hadoop.fs.s3a.aws.credentials.provider", "com.amazonaws.auth.DefaultAWSCredentialsProviderChain") \
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
    .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.1") \
    .getOrCreate()

# here i am counting the number of times each class appears in the test dataset
fractal_data = spark.read.parquet("s3a://ubs-datasets/FRACTAL/data/TEST*.parquet")
lc = fractal_data.groupBy("classification").count()
lc = lc.withColumn('percentage', (lc['count'] / fractal_data.count()) * 100)
lc.show()
# here i am counting the number of times each class appears in the train dataset
fractal_data = spark.read.parquet("s3a://ubs-datasets/FRACTAL/data/TRAIN*.parquet")
lc = fractal_data.groupBy("classification").count()
lc = lc.withColumn('percentage', (lc['count'] / fractal_data.count()) * 100)
lc.show()
# here i am counting the number of times each class appears in the val dataset
fractal_data = spark.read.parquet("s3a://ubs-datasets/FRACTAL/data/VAL*.parquet")
lc = fractal_data.groupBy("classification").count()
lc = lc.withColumn('percentage', (lc['count'] / fractal_data.count()) * 100)
lc.show()