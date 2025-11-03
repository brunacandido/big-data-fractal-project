## Land Cover Classification using Spark - FRACTAL dataset
### Project Overview
This repository contains the implementation of a scalable machine learning pipeline for land cover classification using the FRACTAL dataset. The primary goal is to demonstrate the ability to process and train ML models efficiently on a large point cloud datasets using Apache Spark, MLlib, and AWS cloud infrastructure.

The project includes end-to-end data preprocessing, feature engineering, model training, evaluation, and scalability experiments.

### Dataset
The project uses the FRACTAL dataset, which consists of large-scale LiDAR point clouds in parquet format. More information about the datset can be found here: [Official Dataset Repository](https://huggingface.co/datasets/IGNF/FRACTAL)  

### Project Workflow
1. Data preprocessing
- Handle missing, duplicate, outliers values
- EDA
- Feature engineering
3. Machine Learning
- Train ML model
- Evaluate the model
4. Scalability Experiments
- Evaluate speedup curves for dataset fractions 1%,5%,10%,25%,50%,100% on *n* nodes.

### Repository Structure
*!still in development!
```
big-data-fractal-project/
│
├─ src/                      # PySpark scripts for preprocessing and ML pipeline
│   ├─ preprocess.py         # Data cleaning, filtering, and feature engineering
│   ├─ train_model.py        # Model training and evaluation
│   ├─ scalability.py        # Scripts for running speedup experiments
├─ models/                   # Saved trained models
├─ figures/                  # Plots for report
├─ reports/                  # PDF report describing methodology and results
├─ README.md
└─ requirements.txt          # Python dependencies
```

## Contributors  
[![Contributors](https://contrib.rocks/image?repo=brunacandido/big-data-fractal-project)](https://github.com/brunacandido/big-data-fractal-project/graphs/contributors)
