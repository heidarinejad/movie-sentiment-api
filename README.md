
---

# 🎬 Movie Review Sentiment Classification API

## 📌 Overview

This project implements a binary sentiment classification system for IMDb movie reviews using the Kaggle IMDb 50K Movie Reviews dataset.

The solution includes:

- Exploratory Data Analysis (EDA)
- Text preprocessing and cleaning
- Model development (Logistic Regression & Linear SVM)
- Model evaluation and comparison
- REST API deployment using FastAPI
- Docker containerization

---

## 📊 Dataset

**IMDb Dataset of 50K Movie Reviews**

Source:  
https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

- 50,000 labeled movie reviews
- Balanced dataset (positive / negative)
- Binary classification task

---

## 🔎 Data Analysis

Exploratory analysis includes:

- Class distribution visualization
- Review length distribution
- Text cleaning and normalization
- Data quality checks

Generated plots are saved in:

/reports


---

## 🧠 Model Development

Two machine learning models were trained using TF-IDF features:

### 1. Logistic Regression
- TF-IDF Vectorizer (max_features=20000)
- LogisticRegression (max_iter=1000)

### 2. Linear Support Vector Classifier
- TF-IDF Vectorizer (max_features=20000)
- LinearSVC

Both models were evaluated and compared.

✅ Final selected model for deployment: **Logistic Regression**  
(better performance and supports probability output)

---

## 📈 Model Performance

Accuracy: ~89–90%

Evaluation metrics:

- Confusion Matrix
- ROC Curve (AUC)
- Precision–Recall Curve
- Feature importance visualization

All evaluation artifacts are saved in:

/reports


---

## 🚀 API Usage

The trained model is deployed using FastAPI.

### ▶ Run Locally

```bash
uvicorn app.main:app --reload

Open Swagger documentation:

http://127.0.0.1:8000/docs

```

Endpoint:

POST /predict

Request:
```json
{
  "text": "This movie was amazing!"
}

Response

{
  "prediction": "positive",
  "confidence": 0.9923
}











