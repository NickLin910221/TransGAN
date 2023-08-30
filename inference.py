from Generator import Generator
from Discriminator import Discriminator
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import transforms
import numpy as np
import datetime
import os

time = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

def display(model, input):
    pred = np.squeeze(model(input).detach().cpu().numpy())
    fig = plt.figure(figsize = (4, 4))
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow((pred[i] + 1) / 2)
        plt.axis("off")
    plt.savefig(f"./{time}_inference.png")

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransFormer()

    gen = Generator()
    gen.load_state_dict(torch.load('./model/gen.pt', map_location=device))

    disc = Generator()
    disc.load_state_dict(torch.load('./model.pt', map_location=device))
    
    test_input = torch.randn(16, 28 * 28, device = device)

    display(gen, test_input)