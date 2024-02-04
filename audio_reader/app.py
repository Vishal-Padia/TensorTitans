import gradio as gr
import whisper
from transformers import pipeline
import librosa
import soundfile as sf
import tempfile

model = whisper.load_model("base")

sentiment_analysis = pipeline(
  "sentiment-analysis",
  framework="pt",
  model="SamLowe/roberta-base-go_emotions"
)

def analyze_sentiment(text):
  results = sentiment_analysis(text)
  sentiment_results = {
    result['label']: result['score'] for result in results
  }
  return sentiment_results

def get_sentiment_emoji(sentiment):
  # Define the mapping of sentiments to emojis
  emoji_mapping = {
    "disappointment": "😞",
    "sadness": "😢",
    "annoyance": "😠",
    "neutral": "😐",
    "disapproval": "👎",
    "realization": "😮",
    "nervousness": "😬",
    "approval": "👍",
    "joy": "😄",
    "anger": "😡",
    "embarrassment": "😳",
    "caring": "🤗",
    "remorse": "😔",
    "disgust": "🤢",
    "grief": "😥",
    "confusion": "😕",
    "relief": "😌",
    "desire": "😍",
    "admiration": "😌",
    "optimism": "😊",
    "fear": "😨",
    "love": "❤️",
    "excitement": "🎉",
    "curiosity": "🤔",
    "amusement": "😄",
    "surprise": "😲",
    "gratitude": "🙏",
    "pride": "🦁"
  }
  return emoji_mapping.get(sentiment, "")

def display_sentiment_results(sentiment_results, option):
    sentiment_text = ""
    for sentiment, score in sentiment_results.items():
        emoji = get_sentiment_emoji(sentiment)
        score_percentage = score * 100  # Corrected indentation
        if option == "Sentiment Only":
            sentiment_text += f"{sentiment} {emoji}\n"
        elif option == "Sentiment + Score":
            sentiment_text += f"{sentiment} {emoji}: {score_percentage:.2f}%\n"
    return sentiment_text

def load_and_resample_audio(file_path, target_sample_rate=16000):
    audio, _ = librosa.load(file_path, sr=target_sample_rate)
    temp_file_path = '/tmp/resampled_audio.wav'
    sf.write(temp_file_path, audio, target_sample_rate)
    return temp_file_path

def inference(audio_file_path, sentiment_option):
    resampled_audio_path = load_and_resample_audio(audio_file_path)
    audio = whisper.load_audio(resampled_audio_path)
    audio = whisper.pad_or_trim(audio)

    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    _, probs = model.detect_language(mel)
    lang = max(probs, key=probs.get)

    options = whisper.DecodingOptions(fp16=False)
    result = whisper.decode(model, mel, options)

    sentiment_results = analyze_sentiment(result.text)
    sentiment_output = display_sentiment_results(sentiment_results, sentiment_option)

    return lang.upper(), result.text, sentiment_output

title = "🎤 Gradio UI"
description = "we have deployed our model on Gradio"

block = gr.Blocks()

with block:
    gr.Markdown("# Mood Reader 🕵️‍♂️")
    gr.Markdown("Your Words Whisper 🤫, But Emotions Shout 📢 – Discover What's Truly Behind Every Sentence with Mood Reader 🕵️‍♂️💬")
    with gr.Column():
        audio = gr.Audio(label="Input Audio", type="filepath")
        sentiment_option = gr.Radio(choices=["Sentiment Only", "Sentiment + Score"], label="Select an option")
        transcribe_btn = gr.Button("Transcribe")
    lang_str = gr.Textbox(label="Language")
    text = gr.Textbox(label="Transcription")
    sentiment_output = gr.Textbox(label="Sentiment Analysis Results")

    transcribe_btn.click(
        inference,
        inputs=[audio, sentiment_option],
        outputs=[lang_str, text, sentiment_output]
    )

block.launch()
