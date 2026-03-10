from fastapi import FastAPI, Request
from typing import Any
# ~ import argparse
import joblib
# ~ import uvicorn
from typing import List

model = joblib.load('models/best_estimator_TfidfVectorizer_80000_LogisticRegression.joblib')
vectorizer = joblib.load('models/vectorizer_TfidfVectorizer_100000.pkl')
mlb = joblib.load('models/mlb_100000.pkl')


def prediction(req_data: str) -> List[str]:
    result = model.predict(
        vectorizer.transform([req_data])
    ) # result = array([[0, 1, 1, 0, ...]])
    result = mlb.inverse_transform(result)[0] # result = ('tag1', 'tag2', 'tag3', ...)
    return result


app = FastAPI(description="L'API HTTP pour mon projet OpenClassrooms «Prédiction de l'étiquetage des questions Stackoverflow»", )


@app.post("/predict")
async def predict(req: Request) -> Any:
    req_data = (await req.body()).decode()
    return prediction(req_data)
