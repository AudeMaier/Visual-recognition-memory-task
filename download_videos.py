from pytubefix import YouTube
import os
import pandas as pd

n_videos = 1000

urls = pd.read_csv('sfd.csv')['video_url'].drop_duplicates().tolist()

if not os.path.exists('videos'):
    os.makedirs('videos')

for url in urls:
    if len(os.listdir('videos')) >= n_videos: break
    yt = YouTube(url)
    try:
        print(yt.title)
        mp4_streams = yt.streams.filter(file_extension='mp4')
        mp4_streams[0].download('videos')
    except:
        continue
