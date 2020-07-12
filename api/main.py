from typing import List
import json
from fastapi import FastAPI, Query

from src.models.train import train
from src.models.predict import predict

app = FastAPI()



@app.get("/")
async def test():
    return { 'result': 'hello world' }

@app.get('/train')
async def train_model():
    train(False)

    return { 'result': 'model trained' }


@app.get('/predict')
async def predict_review(reviews: List[str] = Query(..., description='Reviews to process')):

    predictions =  predict(reviews)

    print(predictions)

    return { 'result': predictions }
    # response = [
    #     {
    #         'id': idx + 1,
    #         'review': review,
    #         'prediction': sentiment
    #     }
    #     for idx, (review, sentiment) in enumerate(zip(reviews, predictions))
    # ]

    # return response
