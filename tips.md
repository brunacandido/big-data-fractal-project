# Connecting to clusters
list of clusters  
```
aws emr list-clusters 
``` 

list of active clusters
```
aws emr list-clusters --active
```

to get the id of cluster starting with j
```
aws emr list-clusters --active | grep j
```

to get the information about the cluster i.e. everything inside the cluster
```
aws emr describe-cluster --cluster-id j-3RRKU664DVRTR
```

getting the ec2 instance id
```
aws emr describe-cluster --cluster-id j-3RRKU664DVRTR | grep ec2
```

connecting to ec2 instance i.e. vm
```
ssh ethel@ec2-18-201-99-175.eu-west-1.compute.amazonaws.com
```
*exit : to disconnect from vm*

## CLI STUFF 
moving from windows to Linux
```
cp -r "/mnt/c/Users/Ethel Ogallo/Downloads/shell-lesson-data.zip" ~/
```
where /mnt/c corresponds to C:
we use -r because tis a folder we are sending
use quotes because my user name has a space

```
unzip shell-lesson-data.zip
```

remove/delete directories

```
# if empty
rmdir directory 

# if it has content
rm -r directory 
```

```
ls -F
# a trailing / indicates that this is a directory
# @ indicates a link
# * indicates an executable
```


``` 
touch my_file.txt

# The touch command allows you to efficiently generate a blank text file to be used by such programs.
# By default, mv will not ask for confirmation before overwriting files. However, an additional option, mv -i (or mv --interactive), will cause mv to request such confirmation.
```

## Pyspark installation

**install java**
``` 
java -version #check java version
sudo apt install openjdk-17-jre-headless
```

**install miniconda if not alredy there**
```
 wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

 bash ~/Miniconda3-latest-Linux-x86_64.sh
```
**create and activate conda environment**
```
conda create -n pyspark_env python=3.11 -y
conda activate bd_env
```

**install pyspark**
```
pip install pyspark
```

**work with jupyter lab**
```
pip install ipykernel
python-m ipykernel install --user --name=bd_env --display-name "bd_env"
```

## usage
**check files i.e. count how many total, test/train/eval files**
```
aws s3 ls s3://ubs-datasets/FRACTAL/data/ --recursive | wc -l
aws s3 ls s3://ubs-datasets/FRACTAL/data/ --recursive | grep -i 'train' | wc -l
aws s3 ls s3://ubs-datasets/FRACTAL/data/ --recursive | grep -i 'test' | wc -l
aws s3 ls s3://ubs-datasets/FRACTAL/data/ --recursive | grep -iE 'val|eval' | wc -l
```

## working with Pyspark

start spark session as such 
```
## for the code that will be run in the cluster
spark = SparkSession.builder \
    .appName("ReadDataFromS3") \
    .config("spark.hadoop.fs.s3a.aws.credentials.provider", "com.amazonaws.auth.DefaultAWSCredentialsProviderChain") \
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
    .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.1") \
    .getOrCreate()

```

to run the script in the terminal using local threads
```
spark-submit --master local[2] <yourfile.py>

spark-submit --master local[4] --packages org.apache.hadoop:hadoop-aws:3.3.1 >yourfile.py>
```

to run script on cluster
```
spark-submit --master yarn --num-executor=10 --deploy-mode client wordcount.py
```

to run your code in the cluster save the file in the cluster using teh process
```
# create folder in your home dir
pwd  #check where you are located
mkdir <folder>  #if new folder
nano <file.py>  #create file to host the code. inside paste your code
spark-submit --master yarn <file.py>   # run the code 
```

copying from cluster to s3
```
python3 -m awscli s3 cp fractal.py s3://ubs-homes/erasmus/ethel/
```