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

prompt = f'''I am teaching IELTS reading, writing, speaking and listening course. Here are video subtitle chunks containing audio number which is the exact serial of the video number, the starting time in seconds, the ending time in seconds as well and the text at that time:

{new_df[["id","audio number","start","end","text"]].to_json()}
---------------------------------
"{incoming_query}"
User asked this question related to the video chunks, you have to answer where and how much content is taught in which video and at what timestamps and guide the user to go to that particular video. If user asks unrelated questions, tell him that you can only answer questions related to this course.
'''

with open("C:\Data Science Course\RAG based AI Teaching Assistant\Prompt\prompt.txt", "w") as f:
    f.write(prompt)
# for idx, item in new_df.iterrows():
#     print(idx, item["id"], item["audio number"], item["text"])