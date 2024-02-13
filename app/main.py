from typing import Union

from fastapi import FastAPI, Form

from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

app = FastAPI()

# Load pre-trained tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

class InputData(BaseModel):
    text: str

''''
@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}
'''


@app.post("/predict")
async def predict(data: InputData):
    # Make prediction using pre-trained model
    result = nlp(data.text)

    # Return prediction
    return {"prediction": result[0]['label']}

# LABEL_0 - bad, LABEL_1 - good

    