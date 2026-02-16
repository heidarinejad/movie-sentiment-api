from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os
import re


# Load model (pipeline includes vectorizer)
script_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(script_dir, "../saved_model/sentiment_model.joblib")
model = joblib.load(MODEL_PATH)
print(f"✓ Model loaded from: {MODEL_PATH}")

app = FastAPI(
    title="Movie Review Sentiment API",
    description="Predicts whether a movie review is positive or negative",
    version="1.0"
)

class ReviewRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float

def clean_text(text):
    text = re.sub(r"<.*?>", "", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()

@app.get("/")
def home():
    return {"message": "Movie Sentiment API is running"}

@app.post("/predict", response_model=PredictionResponse)
def predict_sentiment(request: ReviewRequest):
    cleaned = clean_text(request.text)
    prediction = model.predict([cleaned])[0]
    probability= model.predict_proba([cleaned])[0]
    confidence = probability[list(model.classes_).index(prediction)]

    return {
        "prediction": prediction,
        "confidence": round(float(confidence), 4)
    }









