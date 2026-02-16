
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

```

🧾 Response Fields

prediction → Sentiment label (positive or negative)

confidence → Model probability score (between 0 and 1)

# 🐳 Docker Deployment

## 🔨 Build Docker Image

```bash
docker build -t movie-sentiment-api .

```

# ▶ Run Docker Container

```bash
docker run -p 8000:8000 movie-sentiment-api

```

Then access:

```bash
http://localhost:8000/docs

```

## 🧠 Model Training

To retrain the model from scratch:

```bash
python src/train.py

```

The training pipeline will:

Load and clean the dataset

Split training and testing data

Train Logistic Regression and Linear SVM

Compare model performance

Automatically select the best model

Save the final model to /saved_model

Generate evaluation plots in /reports

## 📁 Project Structure

```css
movie-sentiment-api/
│
├── app/
│   ├── main.py
│   └── __init__.py
│
├── src/
│   └── train.py
│
├── dataset/
│   └── IMDB_Dataset.csv
│
├── reports/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── precision_recall_curve.png
│   └── feature_importance.png
│
├── saved_model/
│   └── sentiment_model.joblib
│
├── notebooks/
│   └── sentiment_analysis.ipynb
│
├── Dockerfile
├── requirements.txt
└── README.md

```










