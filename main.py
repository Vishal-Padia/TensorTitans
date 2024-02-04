import gradio as gr
from sent_analysis.sentiment_analysis import predict_sentiment
from speech_to_text.s2t import main as transcribe

def analyze_video(url):
    """
    Takes a YouTube video URL, transcribes it, and analyzes its sentiment.
    """
    transcript = transcribe(url)
    sentiment = predict_sentiment(transcript)
    return sentiment

def app():
    # Define the Gradio interface for Gradio version 3.0 and later
    iface = gr.Interface(
        fn = analyze_video,
        inputs = gr.Textbox(lines=2, placeholder="Enter the URL of the Youtube video:"),
        outputs = gr.Textbox(label="Sentiment"),
        title = "Product Review Analysis",
        description = "This application analyzes the sentiment of a YouTube video review."
    )

    # Launch the app
    iface.launch()

if __name__ == "__main__":
    app()
