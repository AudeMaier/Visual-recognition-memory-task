import os
import torch
from torchvision.models import resnet18, ResNet18_Weights
import torchvision
import argparse
from torchvision import transforms


def process(input_directory: str, output_directory: str):
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    weights = ResNet18_Weights.DEFAULT
    preprocess = weights.transforms()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = resnet18(weights=weights)
    model = torch.nn.Sequential(*(list(model.children())[:-1])).to(device)
    model.eval()

    videos_names = os.listdir(input_directory)

    for video_name in videos_names:
        output_filename = os.path.join(output_directory, video_name[:-4] + '.pt')
        if os.path.exists(output_filename):
            continue
        print(video_name)
        video,_,_ = torchvision.io.read_video(os.path.join(input_directory, video_name), output_format='TCHW', pts_unit='sec', start_pts=0, end_pts=1200)
        print(video.shape)
        if video.shape[0] != 0:

            #split the video in parts of 1000 frames to avoid memory issues and process each part
            i = 0
            tmp_path = os.path.join(output_directory, video_name[:-4])
            if not os.path.exists(tmp_path):
                os.makedirs(tmp_path)
            while len(video) > 0:
                print(i)
                video_part = preprocess(video[:1000].to(device))
                video = video[1000:]
                video_part = model(video_part)

                torch.save(video_part, os.path.join(tmp_path, str(i) + '.pt'))

                i += 1

            #merge all the parts
            files = os.listdir(tmp_path)
            features = []
            for file in files:
                features.append(torch.load(os.path.join(tmp_path, file), weights_only=True))
            features = torch.cat(features)
            for file in files:
                os.remove(os.path.join(tmp_path, file))
            os.rmdir(tmp_path)
            torch.save(features, output_filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess videos')
    # Add arguments with default value "videos" for --input_dir and "video-features" for --output_dir
    parser.add_argument('--input_dir', type=str, default='videos', help='Directory containing videos')
    parser.add_argument('--output_dir', type=str, default='video-features', help='Directory to save preprocessed videos')
    args = parser.parse_args()
    process(args.input_dir, args.output_dir)
