import os
import subprocess
import whisper
import boto3
from transformers import pipeline
import time
import hashlib

# ========== AWS S3 Upload ==========
def upload_to_s3(file_path, bucket_name, s3_key):
    s3 = boto3.client('s3')
    try:
        s3.upload_file(file_path, bucket_name, s3_key)
        print(f"[+] Uploaded {file_path} to s3://{bucket_name}/{s3_key}")
    except Exception as e:
        print("[-] Upload failed:", e)

# ========== Step 1: Download YouTube Video ==========
def download_youtube_video(url, output_path='video.mp4'):
    print("[*] Downloading video...")
    subprocess.run(['yt-dlp', '-f', 'best', '-o', output_path, url], check=True)
    print("[*] Video downloaded.")

# ========== Step 2: Extract Audio ==========
def extract_audio(video_path='video.mp4', audio_path='audio.wav'):
    print("[*] Extracting audio...")
    subprocess.run([
        'ffmpeg', '-y', '-i', video_path, '-vn',
        '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', audio_path
    ], check=True)
    print("[*] Audio extracted.")

# ========== Step 3: Transcribe with Whisper ==========
def transcribe_audio(audio_path='audio.wav'):
    print("[*] Transcribing audio with Whisper...")
    model = whisper.load_model("tiny")
    result = model.transcribe(audio_path)
    transcript = result['text']
    print("[*] Transcription complete.")
    return transcript

# ========== Step 4: QA with DistilBERT ==========
def ask_question(transcript, question):
    print("[*] Answering question...")
    qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
    result = qa_pipeline(question=question, context=transcript)
    return result['answer']

def remove_local_file(path):
    try:
        os.remove(path)
        print(f"[−] Deleted local file: {path}")
    except FileNotFoundError:
        print(f"[!] File not found for deletion: {path}")

# ========== MAIN ==========
if __name__ == "__main__":
    BUCKET_NAME = 'bockmedia'

    youtube_url = input("Enter YouTube URL: ").strip()

    # Generate unique ID based on URL + timestamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    unique_id = hashlib.md5(youtube_url.encode()).hexdigest()[:6] + "-" + timestamp

    video_file = 'video.mp4'
    audio_file = 'audio.wav'
    transcript_file = 'transcript.txt'

    # Download + upload video
    download_youtube_video(youtube_url, video_file)

    extract_audio(video_file, audio_file)

    upload_to_s3(video_file, BUCKET_NAME, f'videos/{unique_id}/{video_file}')
    upload_to_s3(audio_file, BUCKET_NAME, f'audio/{unique_id}/{audio_file}')

    remove_local_file(video_file)

    # Transcribe + upload transcript
    transcript = transcribe_audio(audio_file)
    with open(transcript_file, 'w') as f:
        f.write(transcript)
    upload_to_s3(transcript_file, BUCKET_NAME, f'transcripts/{unique_id}/{transcript_file}')
    remove_local_file(audio_file)
    remove_local_file(transcript_file)

    print("\nTRANSCRIPT PREVIEW:\n", transcript[:500], "...\n")

    # Chat loop
    print("🔁 You can now ask multiple questions about the video. Type 'exit' to quit.\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in ['exit', 'quit']:
            print("👋 Goodbye!")
            break
        answer = ask_question(transcript, question)
        print("Bot:", answer)