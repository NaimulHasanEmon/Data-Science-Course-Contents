import requests

def create_embedding(text):
    r = requests.post("http://localhost:11434/api/embeddings", json={
        "model": "bge-m3",
        "prompt": text
    })

    embedding = r.json()["embedding"]
    return embedding

# Printing first 5 embeddings
print(create_embedding("Cat sat on the mat")[0:5])