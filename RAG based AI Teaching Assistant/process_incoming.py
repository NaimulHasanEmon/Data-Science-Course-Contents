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
    data = r.json()

    # handle both possible keys
    if "embeddings" in data:
        return data["embeddings"]
    elif "embedding" in data:
        return [data["embedding"]]
    else:
        raise ValueError(f"Unexpected response: {data}")

def inference(prompt):
    r = requests.post("http://localhost:11434/api/generate", json={
        # "model": "deepseek-r1",
        "model": "llama3.2",
        "prompt": prompt, 
        "stream": False
    })
    
    response = r.json()
    # print(response)
    return response

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
# print(new_df[["video number", "id","text"]])

prompt = f'''I am teaching IELTS reading, writing, speaking and listening course. Here are video subtitle chunks containing video number, the starting time in seconds, the ending time in seconds as well and the text at that time:

{new_df[["id","video number","start","end","text"]].to_json(orient="records")}
---------------------------------
"{incoming_query}"
User asked this question related to the video chunks, you have to answer in human way 
(don't mention the above format, it's just for you) where and how much content is taught 
in which video and at what timestamps and guide the user to go to that particular video. 
If user asks unrelated questions, tell him that you can only answer questions related to this course. 
You must tell him the video number so that the user can find the video easily, and you must mention 
the starting and ending time so that the user can find the exact time properly. 
All the time must be shown in hour:minutes:seconds format (e.g., 523 seconds -> 0:08:43).

Important rules for answering:
1. The "start" and "end" times are given in seconds. You MUST always convert them to hh:mm:ss format before showing to the user (e.g., 523 -> 00:08:43, 65 -> 00:01:05). Never mention seconds directly. 
2. You must always mention the video number so the user can find it easily. 
3. Always include both the starting and ending times in hh:mm:ss format so the user can locate the exact part. 
4. If the user asks something unrelated to this course, politely say you can only answer questions about this course. 
5. Write your answer in a natural, human way (do not mention JSON or the above format).
'''

with open("C:\Data Science Course\RAG based AI Teaching Assistant\Prompt\prompt.txt", "w") as f:
    f.write(prompt)
# for idx, item in new_df.iterrows():
#     print(idx, item["id"], item["video number"], item["text"])

response = inference(prompt)["response"]
print(response)

with open("C:\Data Science Course\RAG based AI Teaching Assistant\Response/response.txt", "w") as f:
    f.write(response)