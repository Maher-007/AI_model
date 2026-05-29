import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from model.cleandata import clean_data
from model.createmodel import create_model
from model.qa import answer_query

app = FastAPI()

# ==============================
# ✅ LOAD DATA & TRAIN ON START
# ==============================

print(" Loading data...")

# Load dataset
df = pd.read_csv("Data/data.csv")

# Clean data
df = clean_data(df)

# Train model
model = create_model(df)

print(" Model trained and ready!")

# ==============================
# ✅ API MODELS
# ==============================

class Question(BaseModel):
    question: str


# ==============================
# ✅ ROUTES
# ==============================

@app.get("/")
def home():
    return {"message": "AI Model API is running 🚀"}


@app.post("/ask")
def ask_question(q: Question):
    try:
        answer = answer_query(df, q.question)
        return {"answer": answer}
    except Exception as e:
        return {"error": str(e)}


@app.get("/preview")
def preview_data():
    return df.head().to_dict()


@app.get("/stats")
def get_stats():
    return {
        "describe": df.describe().to_dict(),
        "columns": df.columns.tolist()
    }
