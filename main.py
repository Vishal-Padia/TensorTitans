import gradio as gr
from sent_analysis.sentiment_analysis import predict_sentiment
from speech_to_text.s2t import main as transcribe

def inference(text, option):
    """
    Analyzes the sentiment of the input text and returns either the sentiment
    or the sentiment with a score based on the selected option.
    """
    sentiment = predict_sentiment(text)
    if option == "Sentiment Only":
        return sentiment
    else:
        # Assuming predict_sentiment returns a tuple (sentiment, score)
        return f"{sentiment[0]} with a score of {sentiment[1]}"

def analyze_video(url):
    """
    Takes a YouTube video URL, transcribes it, and analyzes its sentiment.
    """
    transcript = transcribe(url)
    sentiment = predict_sentiment(transcript)
    return sentiment

def app():
    block = gr.Blocks()

    with block:
        gr.Markdown("# SentimentScope 🌟💡📊")
        gr.Markdown("### Boost Your Brand - Unravel Customer Sentiments & Propel Product Success 📈💼❤️")
        with gr.Column():
            url_input = gr.Textbox(label="Enter the URL of the YouTube video:", lines=2)
            analyze_btn = gr.Button("Analyze")
        sentiment_output = gr.Textbox(label="Sentiment Analysis Results")

        analyze_btn.click(
            lambda url, option: inference(analyze_video(url), option),
            inputs=[url_input],
            outputs=[sentiment_output]
        )

    block.launch()

if __name__ == "__main__":
    app()
