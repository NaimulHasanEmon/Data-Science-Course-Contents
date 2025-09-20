import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib
import requests

df = joblib.load("C:/Data Science Course/RAG based AI Teaching Assistant/Joblib/embeddings.joblib")

def create_embedding(text_list):
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text_list
    })

    embedding = r.json()["embeddings"]
    return embedding

incoming_query = input("Ask a Question: ")
question_embedding = create_embedding([incoming_query])[0]
# print(question_embedding)

# Find similarities of question_embedding with other embedding
similarities = cosine_similarity(np.vstack(df["embedding"]), [question_embedding]).flatten()
# print(similarities)
top_results = 10
# [::-1] will reverse the array and [0:top_results] will pick first 3 which are max
max_idx = similarities.argsort()[::-1][0:top_results]
# print(max_idx)
new_df = df.loc[max_idx]
# print(new_df[["audio number", "id","text"]])

for idx, item in new_df.iterrows():
    print(idx, item["id"], item["audio number"], item["text"])