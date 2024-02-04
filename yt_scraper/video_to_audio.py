from moviepy.editor import *
from pytube import YouTube
import os


# Define the function to download a YouTube video and convert it to MP3
def download_and_convert_mp4_to_mp3(link):
    # Download the MP4 video
    youtubeObject = YouTube(link)
    youtubeObject = youtubeObject.streams.get_highest_resolution()

    mp4_filename = "../data/video.mp4"
    # mp3_filename = "../data/audio.mp3"
    try:
        # Download the MP4 video and save it in the ../../data/ directory
        mp4_filename = os.path.join("../data", mp4_filename)
        youtubeObject.download(output_path="../data", filename=mp4_filename)
    except:
        print("An error has occurred during download")
        return

    print("Download is completed successfully")

    # Convert the downloaded MP4 to MP3
    mp3_filename = os.path.join("../data/audio.mp3")

    FILETOCONVERT = AudioFileClip(mp4_filename)
    FILETOCONVERT.write_audiofile(mp3_filename)
    FILETOCONVERT.close()


# if __name__ == "__main__":
#     youtube_url = input("Enter the URL of the YouTube video you want to download and convert: ")
#
#     download_and_convert_mp4_to_mp3(youtube_url)