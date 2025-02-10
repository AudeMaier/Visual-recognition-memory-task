# Visual Recognition Memory Task

This goal of this project is to train a recurrent neural netowrk to memorize and recognize frames from videos. The task is designed to mimic the experimental procedure of [Zheng et al. - 2022](https://doi.org/10.1038/s41593-022-01020-w). It consists in first presenting a sequence of short clips to the network, and then testing its ability to classify frames as either "old" (if the frame was part of the clips) or "new" (if the frame was not seen before). The dataset of videos used to generate the clips and the test frames is the [Short Film Dataset - Ghermi et al. - 2024](https://arxiv.org/abs/2406.10221).

The videos first need to be preprocessed by feeding each of their frames through a pretrained CNN to extract features. This preprocessing step is done by the `preprocess_videos.py` script. The resulting features can be downloaded from [this link](...). It can be run as follows:

```bash
python preprocess_videos.py --input_dir path/to/videos --output_dir path/to/save/features
```

The main script to train and test the network is `train.py`. All parameters can be set in a YAML configuration file, which is passed as an argument to the script. An example configuration file is provided in `config-files/config.yaml`. The script can be run as follows:

```bash
python train.py --config path/to/config.yaml
```

The recurrent neural network can be chosen to be an lstm, an improved version of the lstm incorporating a refined gating mechanism ([Gu et al. - 2019](https://arxiv.org/abs/1910.09890)), or a mamba network. Several layers of them can be stacked.

The file `ur_lstm.py` contains the implementation of the improved lstm, it was taken from [this repository](https://gist.github.com/abhshkdz/185f6babd3858fa7c5f0bc986bbca767).