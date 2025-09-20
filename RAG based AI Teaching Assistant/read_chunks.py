import requests
import json
import os
import pprint
import pandas as pd

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
    embeddings = create_embedding([c["text"] for c in content["chunks"]])

    for i, chunk in enumerate(content["chunks"]):
        # print(chunk)
        chunk["embedding"] = embeddings[i]
        my_dict.append(chunk)
    break

# print(my_dict)
pprint.pprint(my_dict)  # pretty print full list

# print(create_embedding(["Cat sat on the mat", "Emon dances on a mat"]))

# pprint.pprint((create_embedding(["Cat sat on the mat", "Emon dances on a mat"])))  # pretty print full list