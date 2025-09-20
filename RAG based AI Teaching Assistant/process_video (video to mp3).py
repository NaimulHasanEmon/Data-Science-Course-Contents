# Converts all the videos to mp3
import os
import subprocess

files = os.listdir("C:\Data Science Course\RAG based AI Teaching Assistant\Videos")

for file in files:
    print(file)
    file_name = file.split(".")[0]
    subprocess.run(["ffmpeg", "-i", f"C:\Data Science Course\RAG based AI Teaching Assistant\Videos/{file}", f"C:\Data Science Course\RAG based AI Teaching Assistant\Audios/{file_name}.mp3"])