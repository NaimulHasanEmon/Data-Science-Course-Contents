# 🎓 RAG-Based AI Teaching Assistant

> Transform your educational videos into an intelligent, searchable knowledge base with AI-powered question answering!

## 🌟 Overview

This project creates a **Retrieval-Augmented Generation (RAG)** system that converts educational videos into an interactive AI teaching assistant. Students can ask questions about course content and get precise answers with exact video timestamps and locations.

### ✨ Key Features

- 🎥 **Video Processing Pipeline**: Automatically converts videos → audio → transcripts → searchable chunks
- 🧠 **AI-Powered Search**: Uses BGE-M3 embeddings for semantic similarity matching
- 🎯 **Precise Answers**: Provides exact video numbers and timestamps (HH:MM:SS format)
- 🌐 **Multi-language Support**: Supports Bengali language transcription with English translation
- 📊 **Efficient Storage**: Uses Joblib for fast embedding storage and retrieval

## 🏗️ Project Architecture

```
RAG based AI Teaching Assistant/
├── 📁 Videos/          # Original video files (MP4)
├── 📁 Audios/          # Extracted audio files (MP3)
├── 📁 JSON's/          # Transcribed text chunks with metadata
├── 📁 Joblib/          # Processed embeddings for fast retrieval
├── 📁 Prompt/          # Generated prompts for LLM
├── 📁 Response/        # AI responses to user queries
└── 🐍 Python Scripts   # Processing pipeline scripts
```

## 🚀 Quick Start Guide

### Prerequisites

- Python 3.8+
- Ollama server running locally (port 11434)
- FFmpeg for video processing
- Required Python packages: `whisper`, `scikit-learn`, `pandas`, `numpy`, `joblib`, `requests`

### Installation

```bash
# Install required packages
pip install openai-whisper scikit-learn pandas numpy joblib requests

# Install FFmpeg (Windows)
# Download from https://ffmpeg.org/download.html

# Start Ollama server with required models
ollama pull bge-m3
ollama pull llama3.2
```

## 📋 Step-by-Step Workflow

### 🎬 Step 1: Collect Your Videos
Place all your educational video files in the `Videos/` folder
- Supported formats: MP4, AVI, MOV, etc.
- Recommended: Clear audio quality for better transcription

### 🎵 Step 2: Convert Videos to Audio
```bash
python "process_video (video to mp3).py"
```
- Extracts high-quality MP3 audio from all videos
- Uses FFmpeg for efficient conversion
- Saves audio files in `Audios/` folder

### 📝 Step 3: Transcribe Audio to Text
```bash
python "create_chunks (mp3 to json).py"
```
- Uses Whisper Large-v2 model for accurate transcription
- Supports Bengali language with English translation
- Creates timestamped text chunks
- Saves structured JSON files in `JSON's/` folder

### 🧮 Step 4: Generate Embeddings
```bash
python "preprocess_json (generate embeddings).py"
```
- Converts text chunks to BGE-M3 embeddings
- Creates searchable vector database
- Saves optimized DataFrame in `Joblib/embeddings.joblib`

### 💬 Step 5: Interactive Q&A
```bash
python process_incoming.py
```
- Ask questions about your course content
- Get AI-powered answers with exact video references
- Includes precise timestamps for easy navigation

## 🎯 Example Usage

**User Question:** "How do you handle missing data in pandas?"

**AI Response:** 
> "Missing data handling in pandas is covered in Video 5 from 00:12:30 to 00:15:45. The instructor explains three main methods: using dropna() to remove missing values, fillna() to replace them with specific values, and interpolate() for numerical data. You can also find additional examples about forward fill and backward fill techniques in Video 5 from 00:16:00 to 00:18:20."

## 🛠️ Technical Details

### Models Used
- **Whisper Large-v2**: For accurate speech-to-text transcription
- **BGE-M3**: For generating high-quality text embeddings
- **Llama 3.2**: For generating natural language responses

### Data Flow
1. **Video** → FFmpeg → **Audio (MP3)**
2. **Audio** → Whisper → **Transcribed Text Chunks**
3. **Text Chunks** → BGE-M3 → **Vector Embeddings**
4. **User Query** → Similarity Search → **Relevant Chunks**
5. **Relevant Chunks** → Llama 3.2 → **Contextual Answer**

## 📊 Performance Features

- **Fast Retrieval**: Cosine similarity search through vector embeddings
- **Efficient Storage**: Joblib serialization for quick loading
- **Scalable**: Handles multiple hours of video content
- **Accurate**: Provides exact timestamps and video references

## 🔧 Customization Options

### Modify Language Settings
Edit `create_chunks (mp3 to json).py`:
```python
result = model.transcribe(
    audio=f"path/to/audio/{audio}", 
    language='en',  # Change to your language
    task="translate"  # or "transcribe"
)
```

### Adjust Search Results
Edit `process_incoming.py`:
```python
top_results = 10  # Change number of relevant chunks
```

### Change AI Models
Update model names in the respective scripts:
- Whisper: `whisper.load_model("base")` for faster processing
- Embeddings: Change `"bge-m3"` to other embedding models
- LLM: Replace `"llama3.2"` with your preferred model

## 📁 File Structure Details

| File | Purpose |
|------|---------|
| `process_video (video to mp3).py` | Converts videos to audio using FFmpeg |
| `create_chunks (mp3 to json).py` | Transcribes audio and creates timestamped chunks |
| `preprocess_json (generate embeddings).py` | Generates vector embeddings for all text chunks |
| `process_incoming.py` | Main interface for asking questions and getting answers |

## 🎓 Perfect For

- **Educational Institutions**: Create searchable course libraries
- **Online Educators**: Help students find specific topics quickly
- **Training Programs**: Build interactive knowledge bases
- **Content Creators**: Make long-form content more accessible

## 🤝 Contributing

Feel free to contribute by:
- Adding support for more languages
- Improving the embedding models
- Enhancing the user interface
- Adding batch processing features

## 📄 License

This project is open-source and available for educational and research purposes.

---

*Transform your educational content into an intelligent, searchable knowledge base today!*