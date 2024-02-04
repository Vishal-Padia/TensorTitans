# Product Review Analysis


## Introduction
This project provides a web application designed to analyze the sentiment of YouTube video content. By leveraging advanced natural language processing (NLP) and machine learning techniques, it transcribes YouTube videos and evaluates the overall sentiment of the spoken content. This tool can be particularly useful for gauging public opinion on products, services, or any topics covered in YouTube videos.

## Features

- Transcription of YouTube video audio to text.
- Sentiment analysis of the transcribed text to determine the overall sentiment (positive, neutral, negative).
- User-friendly web interface for easy interaction with the application.

## Installation
Before running the application, ensure you have Python installed on your system. This application is developed and tested with Python 3.8+. You will also need to install the required dependencies.

1. Clone the repository
```shell
git clone https://github.com/Vishal-Padia/TensorTitans.git
cd TensorTitans
```
2. Create a virtual environment
```shell
python3 -m venv env
```
3. Activate the virtual environment
```shell
source env/bin/activate # Linux/Mac
.\env\Scripts\activate # Windows
```

4. Install required Python packages:
```shell
pip install -r requirements.txt
```

## Usage
To start the application, run the following command from the terminal:
```shell
python main.py
```

Once the application is running, navigate to the displayed URL (typically http://127.0.0.1:7860/) in your web browser. Enter the URL of the YouTube video you wish to analyze in the provided textbox and submit it. The application will process the video, transcribe its audio, perform sentiment analysis on the transcript, and display the sentiment result.

## Dependencies
- `gradio` for creating the web interface.
- `torch` and `torchaudio` for audio processing and transcription.
- Custom modules for sentiment analysis and speech-to-text functionality. 

Ensure all dependencies are correctly installed as per the requirements.txt file to avoid any runtime errors.

## Contribution
We welcome contributions from the community. If you wish to contribute, please fork the repository, make your changes, and create a pull request with a description of your modifications.