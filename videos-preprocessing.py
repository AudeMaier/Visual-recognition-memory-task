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
    
    for video_name in videos_names[103:]:
        video,_,_ = torchvision.io.read_video(os.path.join(input_directory, video_name), output_format='TCHW', pts_unit='sec', start_pts=0, end_pts=1200)
        print(video.shape)
        if video.shape[0] != 0:

            #split the video in parts of 1000 frames to avoid memory issues and process each part
            i = 0
            while len(video) > 0:
                print(i)
                video_part = preprocess(video[:1000].to(device))
                video = video[1000:]
                video_part = model(video_part)

                if not os.path.exists('video-features/'+video_name[:-4]):
                    os.makedirs('video-features/'+video_name[:-4])
                torch.save(video_part, os.path.join(output_directory, video_name[:-4] + '/' + str(i) + '.pt'))

                i += 1

            #merge all the parts
            files = os.listdir('video-features/'+video_name[:-4])
            features = []
            for i in range(len(files)):
                features.append(torch.load(f'video-features/{video_name[:-4]}/{i}.pt', weights_only=True))
            features = torch.cat(features)
            for file in files:
                os.remove('video-features/'+video_name[:-4]+'/'+file)
            os.rmdir('video-features/'+video_name[:-4])
            torch.save(features, os.path.join(output_directory, video_name[:-4] + '.pt'))

                
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess videos')
    parser.add_argument('--input_dir', type=str, help='Directory containing videos')
    parser.add_argument('--output_dir', type=str, help='Directory to save preprocessed videos')
    args = parser.parse_args()
    process(args.input_dir, args.output_dir)