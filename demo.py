import os
import yaml
import torch
import numpy as np
import torch.nn.functional as F 

from torch.utils.data import DataLoader
from torch.optim import Adam

from torchvision.datasets import FashionMNIST
from torchvision import transforms

from model import AMLConv
from plots import sphere_plot




def train_model(model, data, batch_size = 128, max_epochs = 100, learning_rate = 0.001, decay = 1e-5):
    optimizer = Adam(
        model.parameters(), 
        lr = learning_rate,
        weight_decay=decay
    )

    loader = DataLoader(
        dataset = data,
        batch_size = batch_size,
        shuffle = True
    )    

    no_steps = len(loader)   
    model.train()
    for epoch in range(max_epochs):
        for i, (x, labels) in enumerate(loader):            
            err, _ = model(x, labels)
            optimizer.zero_grad()
            err.backward()
            optimizer.step()
                        
            print('Epoch [{}/{}]\t Step [{}/{}]\t Loss: {:.3f}' .format(epoch + 1, max_epochs, i+1, no_steps, err.item()))

    return model


def eval_model(model, data, batch_size = 128):

    loader = DataLoader(
        dataset = data,
        batch_size = batch_size,
        shuffle = True
    )    

    all_labels = []
    all_embeds = []

    model.eval()
    with torch.no_grad():
        for x, labels in loader:
            y = model(x)
            y = F.normalize(y)
            all_labels.append(labels.detach().numpy())
            all_embeds.append(y.detach().numpy())
    
    all_labels = np.concatenate(all_labels)
    all_embeds = np.concatenate(all_embeds)

    return all_embeds, all_labels


def main():
    transformations = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1307,), std=(0.3081,))
        ]
    )

    os.makedirs('./data', exist_ok=True)    
    d = FashionMNIST(
        root = './data',
        train = True,
        transform = transformations,
        download = True
    )
    with open('./config.yml', 'r') as f:
        conf = yaml.safe_load(f) 

    model = AMLConv()
    model = train_model(
        model, 
        d,        
        max_epochs = conf['max_epochs'],
        batch_size = conf['batch_size'],
        learning_rate = conf['learning_rate'],
        decay = conf['decay'],
    )

    embeddings, labels = eval_model(
        model, 
        d, 
        batch_size = conf['batch_size']
    )

    os.makedirs('./results', exist_ok=True)
    sphere_plot(embeddings, labels, figure_path='./results/AMS.png')    


if __name__ == "__main__":   
    main()
