**Movie Review Sentiment Classification API**



*Overview:*
This project builds a sentiment classification model for IMDb movie reviews using the Kaggle IMDb 50K dataset.

The system includes:
* Data analysis and visualization
* Model development (Logistic Regression \& Linear SVM)
* Model evaluation and comparison
* REST API deployment using FastAPI
* Docker containerization

*Dataset:*
IMDb Dataset of 50K Movie Reviews
https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

*Data Analysis:*
* Class distribution visualization
* Review length distribution
* Text cleaning
* Data quality checks

Plots are saved in the /reports folder.

*Model Development:*
Two models were trained:
1. Logistic Regression (TF-IDF features)
2\. Linear Support Vector Classifier
Logistic Regression was selected as the final deployment model.

*Model Performance:*
Accuracy: ~89–90%
Evaluation metrics include:
Confusion Matrix
ROC Curve
Precision-Recall Curve
Feature importance analysis

*API Usage:*
Start locally
'''
$ uvicorn app.main:app --reload
'''
Swagger UI:
http://127.0.0.1:8000/docs

*Endpoint:*
POST /predict
Request:
'''
{
  "text": "This movie was amazing!"
}
'''
Response:
'''
{
  "prediction": "positive",
  "confidence": 0.9923
}
'''
*Docker:*
Build image
'''
$ docker build -t movie-sentiment-api .
'''
Run container:
'''
$ docker run -p 8000:8000 movie-sentiment-api
'''

*Access API:*
http://localhost:8000/docs

Project Structure
'''
movie-sentiment-api/
│
├── app/
│   └── main.py
├── src/
│   └── train.py
├── dataset/
├── reports/
├── notebooks/
├── saved_model/
├── Dockerfile
├── requirements.txt
└── README.md
'''




