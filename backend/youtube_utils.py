import yt_dlp
import requests

# ======================
# 🎥 GET TITLE + THUMBNAIL (NO DOWNLOAD)
# ======================
def get_video_info(url):
    ydl_opts = {
        'quiet': True,
        'skip_download': True
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)

    title = info.get("title", "")
    thumbnail_url = info.get("thumbnail", "")

    return title, thumbnail_url


# ======================
# 🔊 DOWNLOAD AUDIO
# ======================
def download_audio(url):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': 'audio.%(ext)s',
        'quiet': True,

        # 🔥 VERY IMPORTANT
        'ffmpeg_location': 'C:/ffmpeg/bin',

        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }],

        # Stability options
        'nocheckcertificate': True,
        'ignoreerrors': True,
        'retries': 3,
        'fragment_retries': 3,
        'noplaylist': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    return "audio.wav"


# ======================
# 🔗 CLEAN URL
# ======================
def clean_youtube_url(url):
    if "youtu.be" in url:
        video_id = url.split("/")[-1].split("?")[0]
        return f"https://www.youtube.com/watch?v={video_id}"

    if "watch?v=" in url:
        return url.split("&")[0]

    return url


# ======================
# 📸 DOWNLOAD THUMBNAIL
# ======================
def download_thumbnail(url, filename="thumbnail.jpg"):
    response = requests.get(url)

    with open(filename, "wb") as f:
        f.write(response.content)

    return filename