from juliacall import Main as jl
import numpy as np
import os
import argparse

jl.include("generate_dataset.jl")

def save_dataset(data, filename):
    np.savez(filename, clips = data["clips"].to_numpy(),
                       titles = data["titles"].to_numpy(dtype = str))
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create Datasets')
    parser.add_argument('--input_dir', type=str, default='video-features', help='Directory containing the videos')
    parser.add_argument('--videos_tsv', type=str, default='videos.tsv', help='TSV file containing the video names and lengths')
    parser.add_argument('--output_dir', type=str, default='video-features', help='Directory to save the features')
    parser.add_argument('--n_samples_test', type=int, default=2**12, help='Number of samples in the test set')
    parser.add_argument('--n_samples_train', type=int, default=2**14, help='Number of samples in the training set')
    parser.add_argument('--seed', type=int, default=0, help='Seed for the random number generator')
    parser.add_argument('--suffix', type=str, default="", help='Suffix to add to the dataset index files')
    parser.add_argument('--test_set_fraction', type=float, default=.25, help='Fraction of videos used for test set')
    args = parser.parse_args()
    video_tsv = os.path.join(args.input_dir, args.videos_tsv)
    print("Generating test set")
    test_set = jl.create_dataset(videos_tsv = video_tsv, n_samples = args.n_samples_test, fraction_videos = args.test_set_fraction, seed = args.seed)
    save_dataset(test_set, os.path.join(args.output_dir, f"test_indices{args.suffix}.npz"))
    print("Generating training set")
    train_set = jl.create_dataset(videos_tsv = video_tsv, n_samples = args.n_samples_train, excluded_videos = test_set["titles"], seed = args.seed)
    save_dataset(train_set, os.path.join(args.output_dir, f"train_indices{args.suffix}.npz"))
