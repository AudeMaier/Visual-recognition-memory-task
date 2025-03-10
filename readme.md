# Visual Recognition Memory Task

This goal of this project is to train machine learning models to memorize and recognize frames from videos. The task is inspired by the experimental procedure of [Zheng et al. - 2022](https://doi.org/10.1038/s41593-022-01020-w). It consists in first presenting a sequence of short clips (90 clips of approximately 10 seconds duration), and then testing the model's ability to classify frames as either "old" (if the frame was part of the clips) or "new" (if the frame was not seen before). The dataset of videos used to generate the clips and the test frames is the [Short Film Dataset - Ghermi et al. - 2024](https://arxiv.org/abs/2406.10221). The file `sfd.csv` was downloaded [here](https://github.com/ridouaneg/sf20k/blob/main/data/sfd.csv).

## Download Videos

```bash
python download_videos.py
```

## Preprocess Videos

Videos are first preprocessed by feeding each of their frames to a pretrained convolutional neural network (pytorch's resnet18 by default) to extract 512 dimensional feature vectors for each frame.

```bash
python preprocess_videos.py
```

## Training Models

The main script to train and test the network is `train.py`. All parameters can be set in a YAML configuration file, which is passed as an argument to the script.

```bash
python train.py --config config-files/config-example.yaml
```

The recurrent neural network can be chosen to be an lstm, an improved version of the lstm incorporating a refined gating mechanism ([Gu et al. - 2019](https://arxiv.org/abs/1910.09890)), or a mamba network. Several layers of them can be stacked.

The file `ur_lstm.py` contains the implementation of the improved lstm, it was taken from [this repository](https://gist.github.com/abhshkdz/185f6babd3858fa7c5f0bc986bbca767).
