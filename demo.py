import os
import torch

from torch.utils.data import DataLoader
from torch.optim import Adam

from torchvision.datasets import FashionMNIST
from torchvision import transforms

from model import AMLConv
from plots import sphere_plot




def train_model(model, data, batch_size = 128, max_epochs = 100, learning_rate = 0.001, decay = 1e-5, verbosity = 10):
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
        for i, (x, y) in enumerate(loader):
            err, _ = model(x, y)
            optimizer.zero_grad()
            err.backward()
            optimizer.step()
            
            if i % verbosity == 0:
                print('Epoch [{}/{}],\t Step [{}/{}],\t Loss: {:.3f}' .format(epoch + 1, max_epochs, i+1, no_steps, err.item()))

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
        for x, y in loader:
            _, y_ = model(x, y)
            all_labels.append(y.detach().numpy())
            all_embeds.append(y_.detach().numpy())
    
    all_labels = np.concatenate(all_labels)
    all_embeds = np.concatenate(all_embeds)

    return all_embeds, all_labels


def main(args):
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

    model = AMLConv()
    model = train_model(
        model, 
        d,
        max_epochs = args.max_epochs,
        batch_size = args.batch_size,
        learning_rate = args.learning_rate,
        decay = args.decay,
        verbosity = args.verbosity
    )

    embeddings, labels = eval_model(
        model, 
        d, 
        args.batch_size
    )

    os.makedirs('./results', exist_ok=True)
    sphere_plot(embeddings, labels, fig_path='./results/AMS.png')    
