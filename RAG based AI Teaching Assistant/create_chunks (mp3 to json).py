import whisper
import json
import os

model = whisper.load_model("large-v2")

audios = os.listdir("RAG based AI Teaching Assistant/Audios")

for audio in audios:
    print(audio)
    audio_number = audio.split(".")[0]

    result = model.transcribe(audio= f"C:\Data Science Course\RAG based AI Teaching Assistant\Audios\{audio}", language='bn', task="translate")

    segment_arr = result["segments"]
    chunks = []
    for segment in segment_arr:
        chunks.append({"id": segment["id"], "video number": audio_number, "start": segment["start"], "end": segment["end"], "text": segment["text"]})
    
    chunks_with_metadata = {"chunks": chunks, "text": result["text"]}

    # Making json out of it
    with open(f"C:\Data Science Course\RAG based AI Teaching Assistant\JSON's\{audio_number}.json", "w") as f:
        json.dump(chunks_with_metadata, f)