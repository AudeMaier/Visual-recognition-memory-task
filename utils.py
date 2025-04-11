import torch
from torch.utils.data import Dataset
import os

class CustomDataset(Dataset):
    def __init__(self, clips, titles, videos_dir: str, repeat_test: int = 1, subsample: int = 1, device: str = 'cpu', mmap: bool = True):
        self.clips = clips
        self.titles = titles
        self.device = device
        self.subsample = subsample
        self.repeat_test = repeat_test
        self.videos_dir = videos_dir
        self.videos = [self.load(i, mmap = mmap) for i in range(len(titles))]

    def load(self, video_idx, mmap = True):
        return torch.load(os.path.join(self.videos_dir,
                                       f"{self.titles[video_idx][0:-4]}.pt"),
                          mmap = mmap, weights_only=True, map_location = self.device)

    def __len__(self):
        return self.clips.shape[0]

    def __getitem__(self, idx):
        clip = self.clips[idx]
        start = clip[:,0]
        end = clip[:,1]
        video_index = clip[:,2]
        labels = clip[:,3]
        inp = torch.cat([self.videos[video_idx][start_idx:end_idx:self.subsample].repeat_interleave(self.repeat_test if start_idx + 1 == end_idx else 1, dim = 0)
                         for video_idx, start_idx, end_idx in zip(video_index, start, end)]).detach()
        # input, label
        return inp, torch.tensor(labels[labels != -1], dtype = torch.float32)
