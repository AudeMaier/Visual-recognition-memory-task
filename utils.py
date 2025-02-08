import torch
import torch.nn as nn
import random
from torch.utils.data import Dataset
import os
import numpy as np
from datetime import datetime
from ur_lstm import UR_LSTM
from mamba_ssm import Mamba2



def multi_arange(a):
    steps = a[:,2]
    lens = ((a[:,1]-a[:,0]) + steps-np.sign(steps))//steps
    b = np.repeat(steps, lens)
    ends = (lens-1)*steps + a[:,0]
    b[0] = a[0,0]
    b[lens[:-1].cumsum()] = a[1:,0] - ends[:-1]
    return b.cumsum()

class Memorizer(nn.Module):
    def __init__(self, model='urlstm', n_blocks=1, device='cpu'):
        super().__init__()
        self.feature_dim = 512
        self.dim = 513
        self.memory_dim = 512
        self.model = model

        self.linear1 = nn.Linear(self.dim,self.memory_dim)
        
        self.memory_network = []
        assert model in ['lstm', 'urlstm', 'mamba']
        for _ in range(n_blocks):
            if model == 'lstm': self.memory_network.append(nn.LSTM(
                self.memory_dim,
                self.memory_dim,
                num_layers=1,
                batch_first=True
                ).to(device))
            if model == 'urlstm': self.memory_network.append(UR_LSTM(
                self.memory_dim,
                self.memory_dim,
                num_layers=1
                ).to(device))
            if model == 'mamba': self.memory_network.append(Mamba2(
                    d_model=self.memory_dim,
                    d_state=64,
                    d_conv=4,
                    expand=2,
                ).to(device))
        
        self.memory_network = nn.ModuleList(self.memory_network) 
        self.linear2 = nn.Linear(self.memory_dim, 100)
        self.relu = nn.ReLU()
        self.linear3 = nn.Linear(100, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, indices_classifier, device='cpu'):
        x = self.linear1(x.to(device).reshape(x.shape[0]*x.shape[1], x.shape[2])).reshape(x.shape[0], x.shape[1], self.memory_dim)
        for i in range(len(self.memory_network)):
            x = self.memory_network[i](x)
            if self.model!='mamba':
                x = self.relu(x[0])
        y = self.linear2(x[:,indices_classifier,:].reshape(-1, self.memory_dim).to(device))
        y = self.relu(y)
        y = self.linear3(y)
        y = self.sigmoid(y)
        return y.squeeze()


class ClipMemorizer(Memorizer):
    def __init__(self, model='urlstm', n_blocks=1, device='cpu'):
        super().__init__(model, n_blocks, device)
    
    def forward(self, x, device, indices_classifier, batch_size, length_clips, length_tests, input_type = 'train indices'):
        assert input_type in ['train indices', 'test indices', 'clip']
        batch_size, length_clips, length_tests = int(batch_size), int(length_clips), int(length_tests)
        
        if input_type == 'clip': x = x.reshape((1,-1,self.feature_dim))
        elif input_type == 'train indices': x = globals()['videos_train'][x.reshape((-1,))].reshape(batch_size,-1,self.feature_dim)
        elif input_type == 'test indices': x = globals()['videos_test'][x.reshape((-1,))].reshape(batch_size,-1,self.feature_dim)
        
        x = torch.cat((x, torch.cat((torch.ones((batch_size,length_clips,1)),\
                       torch.zeros((batch_size,length_tests,1))), dim=1).to(device)), dim=2)
        return super().forward(x, indices_classifier, device)


    


class CustomClipsDataset(Dataset):
    def __init__(self, videos_dir: str, fps: float, clip_length: tuple, n_videos: tuple,\
                  n_tests: tuple, time_step: int, clip_margin: float, batch_size: int = 1,\
                  test_duration: int = 1, device: str ='cpu', dataset_size: int = 30,\
                  excluded_videos: list = [], curriculum_clips: bool = False):
        self.device = device
        self.videos_dir = videos_dir
        self.titles = [title for title in os.listdir(self.videos_dir) if title.endswith('.pt')\
                        and title not in excluded_videos]

        self.fps = fps
        self.time_step = time_step
        self.dataset_size = dataset_size

        if isinstance(clip_length, int): self.clip_length_max = [clip_length, clip_length+1]
        else:
            assert len(clip_length) == 2 and clip_length[0] < clip_length[1]
            self.clip_length_max = clip_length
        if isinstance(n_videos, int): self.n_videos_max = [n_videos, n_videos+1]
        else:
            assert len(n_videos) == 2 and n_videos[0] < n_videos[1]
            self.n_videos_max = n_videos
        if isinstance(n_tests, int): self.n_tests_max = [n_tests, n_tests+1]
        else:
            assert len(n_tests) == 2 and n_tests[0] < n_tests[1]
            self.n_tests_max = n_tests
        
        self.curriculum_clips = curriculum_clips
        if self.curriculum_clips:
            self.n_videos = [1,2]
            self.clip_length = [2,6]
            self.n_tests = self.n_videos
        else:
            self.n_videos = self.n_videos_max
            self.clip_length = self.clip_length_max
            self.n_tests = self.n_tests_max

        self.clip_margin = clip_margin
        self.batch_size = batch_size
        self.test_duration = test_duration

        self.i = 0

    def load_videos(self, titles, global_name: str = 'videos_train', time_step: int = 1, feature_dim: int = 512):
        titles = [os.path.join(self.videos_dir, title) for title in titles]

        videos = []
        for title in titles:
            videos.append(torch.load(title, map_location=self.device, weights_only=True)\
                          .reshape(-1,feature_dim)[::time_step])
            print(title)
            
        globals()[global_name] = torch.cat(videos)
        return len(globals()[global_name])
    
    def curriculum_clips_update(self):
        if self.n_videos[0] < 40: self.n_videos = [min(self.n_videos_max[0],self.n_videos[0]+2),\
                                                    min(self.n_videos_max[1],self.n_videos[1]+2)]
        else: self.n_videos = [min(self.n_videos_max[0],self.n_videos[0]+3),\
                                min(self.n_videos_max[1],self.n_videos[1]+3)]
        self.clip_length = [min(self.clip_length_max[0], self.clip_length[0]+1),\
                                min(self.clip_length_max[1],self.clip_length[1]+1)]
        self.n_tests = [min(self.n_tests_max[0],self.n_videos[0]),\
                            min(self.n_tests_max[1],self.n_videos[1])]
        
    def __len__(self):
        return 10000000
    

    def __getitem__(self, time_step: int = 1):
        if self.i % self.batch_size == 0:
            self.n_videos_batch = np.random.randint(self.n_videos[0], self.n_videos[1])
            self.n_tests_batch = np.random.randint(self.n_tests[0], self.n_tests[1])*2
            self.clip_length_batch = np.random.randint(self.clip_length[0], self.clip_length[1])

        starts = np.random.randint(0, self.length-self.clip_length_batch*self.fps, size=self.n_videos_batch)
        ends = starts + self.clip_length_batch*self.fps

        seen_margins = multi_arange(np.column_stack(((starts-self.clip_margin*self.fps).clip(min=0),\
                                                      (ends+self.clip_margin*self.fps).clip(max=self.length),\
                                                      np.ones((self.n_videos_batch,), dtype=int))))
        seen = multi_arange(np.column_stack((starts, ends, np.ones((self.n_videos_batch,), dtype=int)*time_step)))

        unseen = np.setdiff1d(multi_arange(np.column_stack(((starts-5*60*self.fps).clip(min=0),\
                                                            (ends+5*60*self.fps).clip(max=self.length),\
                                                            np.ones((self.n_videos_batch,), dtype=int)))), seen_margins)
        
        test_images = np.concatenate((np.random.choice(seen, int(self.n_tests_batch/2)), np.random.choice(unseen, int(self.n_tests_batch/2))))
        
    
        labels = torch.cat([torch.ones(int(self.n_tests_batch/2)), torch.zeros(int(self.n_tests_batch/2))])
        indices = torch.randperm(self.n_tests_batch)
        test_images = test_images[indices]
        labels = labels[indices].squeeze().to(self.device)

        
        test_images = test_images.reshape(-1,1).repeat(self.test_duration, axis=1).reshape(-1)
        
        clip = np.concatenate([seen, test_images])
        indices_classifier = list(len(seen) + (np.arange(self.n_tests_batch)+1) * self.test_duration - 1)
        

        self.i += 1
        return clip, labels, indices_classifier, len(seen), len(test_images)

class TrainingClipDataset(CustomClipsDataset):
    def __init__(self, videos_dir: str, fps: float, clip_length: tuple, n_videos: tuple,\
                  n_tests: tuple, time_step: int, clip_margin: float, batch_size: int = 1,\
                  test_duration: int = 1, device: str ='cpu', curriculum_dataset: bool = False,\
                  curriculum_clips: bool = False, dataset_size: int = 30, excluded_videos: list = []):
        assert fps % time_step == 0
        super().__init__(videos_dir, fps, clip_length, n_videos, n_tests, time_step, clip_margin,\
                          batch_size, test_duration, device, dataset_size, excluded_videos, curriculum_clips)
        titles = np.random.choice(self.titles, size=self.dataset_size, replace=False)
        self.length_max = self.load_videos(titles = titles, global_name='videos_train', time_step=1)
        
        self.curriculum_dataset = curriculum_dataset
        if self.curriculum_dataset: self.length = self.length_max//6
        else: self.length = self.length_max
        

    def __getitem__(self, idx):
        if self.i%15000 == 0 and self.i != 0:
                if self.curriculum_dataset:
                    if self.length != self.length_max:
                        self.length = min(self.length_max, self.length + self.length_max//6)
                    else:
                        titles = np.random.choice(self.titles, size=self.dataset_size, replace=False)
                        self.length_max = self.load_videos(titles=titles, global_name='videos_train',\
                                                        time_step=1)
                        self.length = self.length_max

                if self.curriculum_clips: self.curriculum_clips_update()
                    
                print('train', self.i, self.length, self.n_videos, self.clip_length, self.n_tests)

        return super().__getitem__(time_step=self.time_step)
    
class TestClipDataset(CustomClipsDataset):
    def __init__(self, videos_dir: str, fps: float, clip_length: tuple, n_videos: tuple,\
                  n_tests: tuple, time_step: int, clip_margin: float, batch_size: int = 1,\
                  test_duration: int = 1, device: str ='cpu', curriculum_clips: bool = True,\
                  dataset_size: int = 15):
        assert fps % time_step == 0
        super().__init__(videos_dir, fps, clip_length, n_videos, n_tests, time_step, clip_margin,\
                          batch_size, test_duration, device, dataset_size, [], curriculum_clips)
        self.titles = np.random.choice(self.titles, size=self.dataset_size, replace=False)
        self.length = self.load_videos(titles=self.titles, global_name='videos_test', time_step=1)

    def __getitem__(self, idx):
        if self.i%1500 == 0 and self.i != 0:
            if self.curriculum_clips: self.curriculum_clips_update()
            print('test', self.i, self.length, self.n_videos, self.clip_length, self.n_tests)

        return super().__getitem__(time_step=self.time_step)
