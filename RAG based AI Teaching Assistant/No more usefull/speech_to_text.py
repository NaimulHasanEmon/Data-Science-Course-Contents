import whisper
import json

model = whisper.load_model("large-v2")

result = model.transcribe(audio= "C:\Data Science Course\RAG based AI Teaching Assistant\Audios\sample.mp3", language='bn', task="translate")

segment_arr = result["segments"]
chunks = []

for segment in segment_arr:
    chunks.append({"id": segment["id"], "start": segment["start"], "end": segment["end"], "text": segment["text"]})

# print(chunks)

# Making json out of it
with open("C:\Data Science Course\RAG based AI Teaching Assistant\output.json", "w") as f:
    json.dump(chunks, f)