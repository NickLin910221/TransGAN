from Generator_with_Transformer import Transformer
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import transforms
import numpy as np
import datetime
import os
from PIL import Image, ImageDraw
from torchvision.utils import save_image
import random

time = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ]
)

transform1 = transforms.Compose(
    [
        transforms.CenterCrop((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ]
)

def tensor2numpy(tensor):
    maximum = tensor.max()
    return np.uint8((tensor + 1) / 2 * 255)

if __name__ == "__main__":

    os.mkdir(f"./inference/{time}")

    device = torch.device('cpu')
    model = Transformer(attention_heads = 4).to(device)
    model.load_state_dict(torch.load('./model/best_gen.pt', map_location=device))

    dataset = torchvision.datasets.MNIST("data", train = True, transform = transform, download = True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = 64, shuffle = True)
    original, _ = next(iter(dataloader))

    save_image(original, f"./inference/{time}/original.png")

    for index, image in enumerate(original):
        _image_ = image.cpu().numpy()
        np_img = tensor2numpy(_image_[0])
        _image_ = Image.fromarray(np_img)
        for x in range(3):
            ImageDraw.Draw(_image_).line((random.randint(0, 28), random.randint(0, 28), random.randint(0, 28), random.randint(0, 28)), fill=int(np_img.min()), width=5)
            original[index] = transform1(_image_)
    
    modify = original.clone().to(device)

    save_image(modify, f"./inference/{time}/modify.png")

    inference = model(modify)

    save_image(inference, f"./inference/{time}/inference.png")