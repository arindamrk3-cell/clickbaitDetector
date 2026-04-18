from moviepy import *

def extract_audio(video_path, output_audio="audio.wav"):
    try:
        video = VideoFileClip(video_path)
        audio = video.audio
        audio.write_audiofile(output_audio)
        return output_audio
    except Exception as e:
        print("Audio extraction error:", e)
        return None
# import subprocess
# import os

# def extract_audio(video_path, output_audio="audio.wav"):
#     try:
#         command = [
#             "ffmpeg",
#             "-i", video_path,
#             "-vn",
#             "-acodec", "pcm_s16le",
#             "-ar", "44100",
#             "-ac", "2",
#             output_audio
#         ]

#         subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

#         return output_audio

#     except Exception as e:
#         print("FFmpeg extraction error:", e)
#         return None