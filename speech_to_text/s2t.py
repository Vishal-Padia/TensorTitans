import torch
import torchaudio
import os
from yt_scraper.video_to_audio import download_and_convert_mp4_to_mp3
from pytube import YouTube
import re

def normalize_title(title):
    # Replace | with a space and remove other non-alphanumeric characters except spaces and dashes
    normalized_title = re.sub(r'[^\w\s-]', '', title.replace('|', ' '))
    # Replace multiple spaces with a single space
    normalized_title = re.sub(r'\s+', ' ', normalized_title)
    # Trim leading and trailing spaces
    normalized_title = normalized_title.strip()
    return normalized_title

def initialize_model(device):
    """
    Initialize the model and set it to the specified device.
    """
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    model = bundle.get_model().to(device)
    return model, bundle

def load_and_preprocess_audio(file_path, device, target_sample_rate):
    """
    Load an audio file, resample it if necessary, and move it to the specified device.
    """
    waveform, sample_rate = torchaudio.load(file_path, normalize=True)
    waveform = waveform.to(device)
    if sample_rate != target_sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, target_sample_rate)
    return waveform

class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emission: torch.Tensor) -> str:
        indices = torch.argmax(emission, dim=-1)
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]
        return "".join([self.labels[i] for i in indices])

def generate_transcript(model, waveform, labels):
    """
    Generate the transcript for the given audio waveform using the specified model.
    """
    with torch.inference_mode():
        emission, _ = model(waveform)
    decoder = GreedyCTCDecoder(labels=labels)
    raw_transcript = decoder(emission[0])
    transcript = raw_transcript.replace("|", " ")
    return transcript

def main(youtube_url):
    """
    Main function to process a YouTube URL for transcription.
    """
    # Configuration and initialization
    torch.random.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model
    model, bundle = initialize_model(device)

    # Download and convert video
    download_and_convert_mp4_to_mp3(youtube_url)

    # Build speech file path
    audio_dir = os.path.join("../data")
    speech_file = os.path.join(audio_dir, f"audio.mp3")

    # Rest of speech pipeline
    waveform = load_and_preprocess_audio(speech_file, device, bundle.sample_rate)
    transcript = generate_transcript(model, waveform, bundle.get_labels())

    with open("../data/transcript.txt", "w") as file:
        file.write(transcript)

    return transcript

# if __name__ == "__main__":
#     youtube_url = input("Enter YouTube URL: ")  # Keep this for direct script execution
#     main(youtube_url)
