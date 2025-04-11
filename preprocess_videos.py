import os
import torch
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import Dataset, DataLoader
import torchvision
import argparse
import time
import numpy as np

def number_of_frames(reader):
    metadata = reader.get_metadata()['video']
    return int(metadata['fps'][0] * metadata['duration'][0])

def write_video_lengths(titles, lengths, output_file: str):
    with open(output_file, 'w') as f:
        for title, length in zip(titles, lengths):
            f.write(f"{title}\t{length}\n")

def video_feature_lengths(titles, features_dir: str):
    lengths = []
    for title in titles:
        lengths.append(torch.load(os.path.join(features_dir, f"{title[0:-4]}.pt"),
                                  mmap = True, weights_only=True,).shape[0])
    return lengths

class VideoDataset(Dataset):
    def __init__(self, input_dir):
        self.input_dir = input_dir
        titles = []
        readers = []
        lenghts = []
        for title in os.listdir(input_dir):
            try:
                reader = torchvision.io.VideoReader(os.path.join(input_dir, title))
                titles.append(title)
                readers.append(reader)
                lenghts.append(number_of_frames(reader)) # approximate lenghts
            except:
                print(f"Error reading video {title}")
        self.titles = titles
        self.readers = readers
        self.lenghts = lenghts
        self.current_reader = 0

    def __len__(self):
        return sum(self.lenghts)

    def __getitem__(self, idx):
        reader = self.readers[self.current_reader]
        try:
            return reader.__next__()['data'], self.titles[self.current_reader]
        except:
            self.current_reader += 1
            return self.__getitem__(idx)


def process(input_directory: str, output_directory: str, batch_size: int = 4*1024):
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    weights = ResNet18_Weights.DEFAULT
    preprocess = weights.transforms()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = resnet18(weights=weights)
    model = torch.nn.Sequential(*(list(model.children())[:-1])).to(device)
    model.eval()

    dataset = VideoDataset(input_directory)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    features = []
    t0 = time.time()
    old_title = dataset.titles[0]
    for idx, (video_segment, title) in enumerate(dataloader):
        with torch.no_grad():
            f = model(preprocess(video_segment.to(device))).reshape(-1, 512).cpu()
        features.append(f)
        if title[0] != title[-1] or idx == len(dataloader) - 1 or old_title != title[0]:
            if len(features) != 0:
                print(f"Saving video {title[0]}. Processing time: {time.time() - t0}")
                if title[0] != title[-1]:
                    features[-1] = f[np.where(np.array(title) == title[0])[0]]
                    new_features = [f[np.where(np.array(title) == title[-1])[0]]]
                else:
                    new_features = []
                video_features = torch.cat(features)
                torch.save(video_features, os.path.join(output_directory, f'{title[0][:-4]}.pt'))
                features = new_features
                t0 = time.time()
        old_title = title[-1]
    lengths = video_feature_lengths(dataset.titles, output_directory)
    video_tsv = os.path.join(output_directory, 'videos.tsv')
    write_video_lengths(dataset.titles, lengths, video_tsv)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess videos')
    # Add arguments with default value "videos" for --input_dir and "video-features" for --output_dir
    parser.add_argument('--input_dir', type=str, default='videos', help='Directory containing videos')
    parser.add_argument('--output_dir', type=str, default='video-features', help='Directory to save preprocessed videos')
    parser.add_argument('--batch_size', type=int, default=4*1024, help='Batch size for processing videos')
    args = parser.parse_args()
    process(args.input_dir, args.output_dir, args.batch_size)
