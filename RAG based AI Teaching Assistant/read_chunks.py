import requests
import json
import os
# import pprint
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def create_embedding(text_list):
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text_list
    })

    embedding = r.json()["embeddings"]
    return embedding


folder = r"C:\Data Science Course\RAG based AI Teaching Assistant\JSON's"
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
    break       # For single json file

# print(my_dict)
# pprint.pprint(my_dict)  # pretty print full list

df = pd.DataFrame.from_records(my_dict)
# print(df)

incoming_query = input("Ask a Question: ")
question_embedding = create_embedding([incoming_query])[0]
# print(question_embedding)

# Find similarities of question_embedding with other embedding
similarities = cosine_similarity(np.vstack(df["embedding"]), [question_embedding]).flatten()
print(similarities)
top_results = 3
# [::-1] will reverse the array and [0:top_results] will pick first 3 which are max
max_idx = similarities.argsort()[::-1][0:top_results]
print(max_idx)
new_df = df.loc[max_idx]
print(new_df[["audio number", "id","text"]])