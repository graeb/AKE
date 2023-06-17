# test with this
# https://www.youtube.com/watch?v=V4qrvoFodmo
import yt_dlp
import os


def download_audio(video_url: str) -> str:
    """_summary_

    Returns:
        str: _description_
    """

    file_path = os.path.join(os.getcwd(), 'audio', 'file.mp3')
    if video_url.find('watch?') == -1:
        video_url = f'https://www.youtube.com/watch?v={video_url}'

    # Set the options for audio extraction
    options = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        # 'outtmpl': './audio/%(title)s.%(ext)s'
        'outtmpl': file_path
    }
    # Download the audio
    with yt_dlp.YoutubeDL(options) as ydl:
        ydl.download([video_url])

    return file_path
