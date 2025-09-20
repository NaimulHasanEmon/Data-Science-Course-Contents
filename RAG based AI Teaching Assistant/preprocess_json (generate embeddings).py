import requests
import json
import os
# import pprint
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib

def create_embedding(text_list):
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text_list
    })

    embedding = r.json()["embeddings"]
    return embedding


folder = r"C:\Data Science Course/RAG based AI Teaching Assistant/JSON's"
jsons = os.listdir(folder)      # List all the json's
my_dict = []

for json_file in jsons:
    file_path = os.path.join(folder, json_file)

    with open(file_path) as f:
        content = json.load(f)
    
    # List comprehension that will give a list of all the text and then I will pass this to create_embedding
    
    print(f"Creating embeddings for {json_file} file")
    embeddings = create_embedding([c["text"] for c in content["chunks"]])

    for i, chunk in enumerate(content["chunks"]):
        # print(chunk)
        chunk["embedding"] = embeddings[i]
        my_dict.append(chunk)
    # break       # For single json file

# print(my_dict)
# pprint.pprint(my_dict)  # pretty print full list

df = pd.DataFrame.from_records(my_dict)
# Save this dataframe
joblib.dump(df, r"C:/Data Science Course/RAG based AI Teaching Assistant/Joblib/embeddings.joblib")
print("Embeddings joblib complete!")