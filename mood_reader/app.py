import gradio as gr
from transformers import pipeline

# Load the sentiment analysis model
sentiment_analysis = pipeline(
  "sentiment-analysis",
  framework="pt",
  model="SamLowe/roberta-base-go_emotions"
)

def analyze_sentiment(text):
  results = sentiment_analysis(text)
  # Sort the results by score in descending order
  sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
  # Select the top 3 labels and their scores
  top_3_labels_scores = {result['label']: result['score'] for result in sorted_results[:3]}
  return top_3_labels_scores

def get_sentiment_emoji(sentiment):
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
        score_percentage = score * 100
        if option == "Sentiment Only":
            sentiment_text += f"{sentiment} {emoji}\n"
        elif option == "Sentiment + Score":
            sentiment_text += f"{sentiment} {emoji}: {score_percentage:.2f}%\n"
    return sentiment_text

def inference(text_input, sentiment_option):
    sentiment_results = analyze_sentiment(text_input)
    sentiment_output = display_sentiment_results(sentiment_results, sentiment_option)

    return sentiment_output

title = "🎤 Gradio UI"
description = "we have deployed our model on Gradio"

block = gr.Blocks()

with block:
    gr.Markdown("#  🕵️")
    gr.Markdown("Between the Lines, Emotions Speak 🤫📖 - Decode the Silent Echoes with Mood Reader 🕵️‍♂️💬 Every Sentence with Mood Reader 🕵️‍♂️💬")
    with gr.Column():
        text_input = gr.Textbox(label="Input Text", lines=4)
        sentiment_option = gr.Radio(choices=["Sentiment Only", "Sentiment + Score"], label="Select an option")
        analyze_btn = gr.Button("Analyze")
    sentiment_output = gr.Textbox(label="Sentiment Analysis Results")

    analyze_btn.click(
        inference,
        inputs=[text_input, sentiment_option],
        outputs=[sentiment_output]
    )

block.launch()
