from utils import CustomDataset
import argparse
from models import TransformerEncoder, xLSTM, LSTM, Model, Mamba
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch import nn
import numpy as np
import wandb
import yaml
import time
import os

def timing(t0, description):
    t1 = time.time()
    print(f"{description} {t1-t0}s")
    return t1


def main(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)

    show_timing = 'show_timing' in config and config['show_timing'] == True

    if config['logging']['wandb']:
        wandb.login(key=os.environ["WANDB_API_KEY"])
        wandb.init(
            project=config['logging']['wandb_project'],
            config=config,
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_indices = np.load(config['dataset']['test_indices_file'])
    train_indices = np.load(config['dataset']['train_indices_file'])

    train_loader = DataLoader(CustomDataset(train_indices['clips'],
                                            train_indices['titles'],
                                            config['dataset']['video_path'],
                                            mmap = config['dataset']['mmap'],
                                            subsample = config['dataset']['subsample'],
                                            repeat_test = config['dataset']['repeat_test']),
                              batch_size = config['training']['batch_size'],
                              num_workers = config['training']['num_workers'])
    test_loader = DataLoader(CustomDataset(test_indices['clips'],
                                           test_indices['titles'],
                                           config['dataset']['video_path'],
                                           mmap = config['dataset']['mmap'],
                                           subsample = config['dataset']['subsample'],
                                           repeat_test = config['dataset']['repeat_test']),
                             batch_size = config['training']['batch_size'],
                             num_workers = config['training']['num_workers'])

    model = Model(eval(config['model']['model']),
                       subsample = config['dataset']['subsample'],
                       repeat_test = config['dataset']['repeat_test'],
                       n_test = 90,
                       **config['model']['args']).to(device)

    if config['model']['load']: model.load_state_dict(torch.load(config['model']['load_path'], weights_only=True))

    loss_function = nn.BCELoss()
    optimizer = AdamW(model.parameters(), lr = config['training']['learning_rate'])

    if not os.path.exists(os.path.dirname(config['logging']['logfile'])):
        os.makedirs(os.path.dirname(config['logging']['logfile'])
    with open(config['logging']['logfile'], 'w') as file:
        file.write('train_step train_loss train_acc test_loss test_acc\n')

    train_acc = 0
    train_loss = 0
    model.train(True)
    for epoch in range(config['training']['n_epochs']):
        train_acc = 0
        train_loss = 0
        print(epoch)
        if show_timing: t0 = time.time()
        for x, y in train_loader:
            if show_timing: t0 = timing(t0, "Loaded data: ")
            y_hat = model(x.to(device))
            if show_timing: t0 = timing(t0, "Forward: ")
            y = y.to(device)
            loss = loss_function(y_hat, y)
            loss.backward()
            if show_timing: t0 = timing(t0, "Backward: ")
            optimizer.step()
            optimizer.zero_grad()
            if show_timing: t0 = timing(t0, "Param Update: ")
            train_acc += (y_hat.round() == y).detach().float().mean()
            train_loss += loss.detach().item()
            if show_timing: t0 = timing(t0, "Train Loss/Acc Update: ")

        train_acc /= len(train_loader)
        train_loss /= len(train_loader)

        model.eval()
        test_loss = 0
        test_acc = 0
        with torch.no_grad():
            for x, y in test_loader:
                y_hat = model(x.to(device))
                y = y.to(device)
                test_loss += loss_function(y_hat, y)
                test_acc += (y_hat.round() == y).float().mean()

        test_acc /= len(test_loader)
        test_loss /= len(test_loader)

        if config['logging']['wandb']:
            wandb.log({'training loss': train_loss, 'training accuracy': train_acc,
                       'test loss': test_loss, 'test accuracy': test_acc})
        with open(config['logging']['logfile'], 'a') as file:
            file.write(f'{epoch} {train_loss} {train_acc} {test_loss} {test_acc}\n')

        model.train(True)
        if config['model']['save']:
            if not os.path.exists(os.path.dirname(config['model']['save_path'])):
                os.makedirs(os.path.dirname(config['model']['save_path']))
            torch.save(model.state_dict(), config['model']['save_path'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--config-file', type=str, help='Path to config file', required=True)

    args = parser.parse_args()
    main(args.config_file)
