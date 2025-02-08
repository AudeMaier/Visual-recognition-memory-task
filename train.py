import argparse
from utils import ClipMemorizer, TrainingClipDataset, TestClipDataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
import yaml

def main(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file) 

    if config['general']['wandb']:
        WANDB_API_KEY=config['general']['wandb_key']
        wandb.login(key=WANDB_API_KEY)
        run = wandb.init(
            project=config['general']['wandb_project'],
            config=config,
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = config['general']['batch_size']

    
    dataset_test = TestClipDataset(batch_size=batch_size, device=device, **config['dataset']['test'])
    dataset = TrainingClipDataset(batch_size=batch_size, device=device, **config['dataset']['train'],\
                                  excluded_videos=dataset_test.titles)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)

    model = ClipMemorizer(model=config['model']['memory_model'], n_blocks=config['model']['n_blocks'], device=device).to(device)
    if config['model']['load']: model.load_state_dict(torch.load(config['model']['load_path'], weights_only=True))
    loss_function = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)


    train_acc = 0
    train_loss = 0
    model.train(True)
    train_step = 0
    for data in dataloader:
        print(train_step)
        inputs, labels, indices_classifier, length_clips, length_tests = data
        optimizer.zero_grad()
        
        outputs = model(inputs, indices_classifier=torch.tensor([i[0] for i in indices_classifier]), \
                        batch_size=batch_size, \
                        length_clips=length_clips[0], \
                        length_tests=length_tests[0], \
                        input_type='train indices', \
                        device=device)
                    
        loss = loss_function(outputs.squeeze(), labels.flatten())
        loss.backward()
        optimizer.step()
        train_acc += (outputs.round().squeeze() == labels.flatten()).float().mean()
        train_loss += loss.item()
        
        if train_step % 100 == 99:
            model.eval()
            test_loss = 0
            test_acc = 0
            with torch.no_grad():
                    for k in range(10):
                        data = next(iter(dataloader_test))
                        inputs, labels, indices_classifier, length_clips, length_tests = data
                        
                        outputs = model(inputs, indices_classifier=torch.tensor([i[0] for i in indices_classifier]), \
                                        batch_size=batch_size, \
                                        length_clips=length_clips[0], \
                                        length_tests=length_tests[0], \
                                        input_type='test indices', \
                                        device=device)
                            
                        test_loss += loss_function(outputs.squeeze(), labels.squeeze().flatten())
                        test_acc += (outputs.round().squeeze() == labels.squeeze().flatten()).float().mean()

            if config['general']['wandb']: 
                wandb.log({'training loss': train_loss/100, 'training accuracy': train_acc/100, 'test loss': test_loss/10,\
                            'test accuracy': test_acc/10})
            
            train_acc = 0
            train_loss = 0
            model.train(True)
            if config['model']['save']: torch.save(model.state_dict(), config['model']['save_path'])
        train_step += 1
    
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--config-file', type=str, help='Path to config file', required=True)
    
    args = parser.parse_args()
    main(args.config_file)