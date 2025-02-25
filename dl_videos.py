from pytubefix import YouTube
import os
import pandas as pd

n_videos = 500

urls = pd.read_csv('sfd.csv')['video_url']

for url in urls:
    if len(os.listdir('videos')) >= n_videos: break
    yt = YouTube(url)
    try:
        mp4_streams = yt.streams.filter(file_extension='mp4')
        mp4_streams[0].download('videos')
    except:
        continue