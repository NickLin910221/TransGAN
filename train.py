from Generator import Generator
from Discriminator import Discriminator
from Transformer import Transformer
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
import datetime
import os
from torch.autograd import Variable
from PIL import Image, ImageDraw
import random

time = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
os.mkdir(f"./train/{time}")

torch.autograd.set_detect_anomaly(True)

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

def display(model, x, epoch):
    save_image(model(x)[:64], f"./train/{time}/inference_epoch_{epoch}.png")
    save_image(x[:64], f"./train/{time}/original_epoch_{epoch}.png")

def loss(G_loss, D_loss):
    if G_loss[-1] < (max(G_loss) / 100) and D_loss[-1] < (max(D_loss) / 100):
        return True

def tensor2numpy(tensor):
    maximum = tensor.max()
    return np.uint8((tensor + 1) / 2 * 255)

if __name__ == "__main__":

    epochs, epoch = 150, 0
    batch_size = 1024
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = torchvision.datasets.MNIST("data", train = True, transform = transform, download = True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = True)

    trans = Transformer(layer = 1, attention_heads = 4, device = device).to(device)
    gen = Generator().to(device)
    disc = Discriminator(batch_size).to(device)

    g_optim = torch.optim.Adam(trans.parameters(), lr = 0.002)
    d_optim = torch.optim.Adam(disc.parameters(), lr = 0.002)

    loss_function_BCE = torch.nn.BCELoss()

    G_loss = []
    D_loss = []

    while not(epoch > epochs) or not loss(G_loss, D_loss):
        gen_epoch_loss = 0
        disc_epoch_loss = 0
        batch_cnt = len(dataloader.dataset)

        for step, (img, _) in enumerate(dataloader):
            original_img = img.clone().to(device)
            for index, image in enumerate(img):
                _image_ = image.cpu().numpy()
                np_img = tensor2numpy(_image_[0])
                _image_ = Image.fromarray(np_img)
                # for x in range(3):
                ImageDraw.Draw(_image_).line((random.randint(0, 28), random.randint(0, 28), random.randint(0, 28), random.randint(0, 28)), fill=int(np_img.min()), width=3)
                image = transform1(_image_)

            if epoch == 0 and step == 0:
                test_input = img.clone().to(device)

            noise_img = img.clone().to(device)
            # Discriminator Train
            disc.zero_grad()
            real_output = disc(original_img)
            d_real_loss = loss_function_BCE(real_output, torch.ones_like(real_output))
            
            gen_img = trans(noise_img)
            fake_output = disc(gen_img)
            d_fake_loss = loss_function_BCE(fake_output, torch.zeros_like(fake_output))

            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            d_optim.step()

            # Generator Train
            trans.zero_grad()

            gen_img = trans(noise_img)
            fake_output = disc(gen_img)
            g_loss = loss_function_BCE(fake_output, torch.ones_like(fake_output))

            g_loss.backward()
            g_optim.step()

            with torch.no_grad():
                disc_epoch_loss += d_loss
                gen_epoch_loss += g_loss

        if epoch % 25 == 0:
            display(trans, test_input, epoch)

        if epoch % 100 == 0:
            torch.save(trans.state_dict(), f"./train/{time}/gen_{epoch}.pt")
            torch.save(disc.state_dict(), f"./train/{time}/disc_{epoch}.pt")
        
        if len(D_loss) == len(G_loss):
            plt.plot([i + 1 for i in range(len(G_loss))], G_loss, color=(255/255, 0/255, 0/255), label = 'Generator')
            plt.plot([i + 1 for i in range(len(D_loss))], D_loss, color=(0/255, 0/255, 255/255), label = 'Discriminator')
            plt.xlabel('Epoch', {'fontsize': 10, 'color':'black'})
            plt.ylabel('Loss', {'fontsize': 10, 'color':'black'})
            plt.legend(loc = 1)
            plt.savefig(f"./train/{time}/loss.png")
            plt.close()

        with torch.no_grad():
            disc_epoch_loss /= batch_cnt
            gen_epoch_loss /= batch_cnt
            D_loss.append(disc_epoch_loss.item())
            G_loss.append(gen_epoch_loss.item())

            if min(D_loss) == disc_epoch_loss.item():
                torch.save(disc.state_dict(), f"./train/{time}/best_disc.pt")
            if min(G_loss) == gen_epoch_loss.item():
                torch.save(gen.state_dict(), f"./train/{time}/best_gen.pt")

            print(f"{epoch} | Generator_loss : {gen_epoch_loss}, Discriminator_loss : {disc_epoch_loss}")
            epoch += 1

    torch.save(gen.state_dict(), f"./model/gen.pt")
    torch.save(disc.state_dict(), f"./model/disc.pt")
    torch.save(gen.state_dict(), f"./train/{time}/gen.pt")
    torch.save(disc.state_dict(), f"./train/{time}/disc.pt")