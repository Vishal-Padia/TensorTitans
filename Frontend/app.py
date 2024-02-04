from flask import Flask, request, render_template
import os
from sent_analysis.sentiment_analysis import predict_sentiment
from speech_to_text.s2t import main as transcribe

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        youtube_url = request.form['youtube_url']
        # Process the YouTube URL (Download, Convert, Transcribe)
        transcript = transcribe(youtube_url)
        print(transcript)
        # Check if transcript is not None and is a string
        if transcript and isinstance(transcript, str):
            sentiment = predict_sentiment(transcript)
        else:
            # Handle the case where transcription failed or returned None
            transcript = "Transcription failed or returned no content."
            sentiment = "N/A"  # Or any default/fallback value
        return render_template('index.html', transcript=transcript, sentiment=sentiment)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)