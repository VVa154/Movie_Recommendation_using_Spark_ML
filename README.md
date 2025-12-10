# 🎬 Movie Recommendation System using Spark ML, Autoencoders & H2O

A complete end-to-end **Movie Recommendation System** built using three different machine learning frameworks:

- **Collaborative Filtering using Apache Spark ALS**
- **Neural Autoencoders (TensorFlow)**
- **Machine Learning Models in H2O (Random Forest, GBM, Deep Learning)**

This project uses the **MovieLens ml-20m dataset**, a widely-used dataset for benchmarking recommender systems.

# 📘 Project Overview

This project builds a movie recommender using multiple ML approaches on the **MovieLens 20M dataset**, which contains:

- **20,000,263 ratings**
- **465,564 tag applications**
- **27,278 movies**
- Data collected from **138,493 users** over **20 years**

Dataset Source:  
👉 https://grouplens.org/datasets/movielens/

---

# 🧠 Approaches Implemented

## 1️⃣ Collaborative Filtering using Spark ALS

**File:** `Spark_ML.py`

This implementation uses the **Alternating Least Squares (ALS)** algorithm from **PySpark MLlib**.

### Features
- Train/test split  
- RMSE evaluation  
- Generate top-10 movie recommendations for all users  
- Handles cold-start predictions  

---

## 2️⃣ Autoencoders (TensorFlow)

**File:** `Auto_Encoders.py`

A neural autoencoder model that reconstructs user-movie rating vectors to predict missing ratings.

### Highlights
- User–movie matrix pivot  
- Feed-forward autoencoder  
- Mean Squared Error loss  
- Predictions for unseen ratings  
- Manual forward pass (no Keras, fully TensorFlow low-level)  

---

## 3️⃣ H2O Machine Learning Models

**File:** `H2O.py`

Models implemented:

- **Random Forest**
- **Gradient Boosting Machine (GBM)**
- **Deep Learning (DL)**
- Weighted Average Ensemble

### Features
- Auto data split (train/valid/test)  
- AUC evaluation  
- Predictions using all models  
- Simple weighted ensemble for better accuracy  

---

# ⚙️ Installation & Setup

### Prerequisites
- Python 3.7+  
- Spark 2.x or 3.x  
- TensorFlow  
- H2O.ai Python client  
- Pandas, NumPy, Matplotlib  
- Java (required for Spark & H2O)

---

## ▶️ How to Run Each Model

---

## **1. Running Spark ALS Model**

Ensure SPARK_HOME is set in your environment.

```bash
python Spark_ML.py

2. Running Autoencoder Model
python Auto_Encoders.py

Requires: 
tensorflow
numpy
pandas
scikit-learn

3. Running H2O Models
python H2O.py

Requires: pip install h2o

Dataset Description
This project uses the official MovieLens ml-20m dataset, which includes:
ratings.csv – userId, movieId, rating, timestamp
movies.csv – movieId, title, genres
links.csv – IMDB & TMDB IDs
genome-tags.csv – tag labels
genome-scores.csv – movie–tag relevance
tags.csv – user-generated tags
Dataset source:
https://grouplens.org/datasets/movielens/

The MovieLens dataset is licensed for research use only and must not be used for:
commercial purposes
redistribution
endorsement claims
Full license text from the dataset README is included in this repository.

