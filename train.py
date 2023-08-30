from Generator import Generator
from Discriminator import Discriminator
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import transforms
import numpy as np
import datetime
import os


from torchvision.utils import save_image

time = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
os.mkdir(f"./train/{time}")

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ]
)

def display(model, input, epoch):
    pred = np.squeeze(model(input).detach().cpu().numpy())
    fig = plt.figure(figsize = (4, 4))
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow((pred[i] + 1) / 2)
        plt.axis("off")
    plt.savefig(f"./train/{time}/epoch_{epoch}.png")

def loss(G_loss, D_loss):
    if G_loss[-1] < (max(G_loss) / 100) and D_loss[-1] < (max(D_loss) / 100):
        return True


if __name__ == "__main__":
    epochs, epoch = 150, 0
    batch_size = 64
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dataset = torchvision.datasets.MNIST("data", train = True, transform = transform, download = True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = True)
    test_input = torch.rand(16, 100, device = device)

    gen = Generator().to(device)
    disc = Discriminator().to(device)

    g_optim = torch.optim.Adam(gen.parameters(), lr = 0.002)
    d_optim = torch.optim.Adam(disc.parameters(), lr = 0.002)

    loss_function = torch.nn.BCELoss()

    G_loss = []
    D_loss = []

    while not(epoch > epochs) or not loss(G_loss, D_loss):
        gen_epoch_loss = 0
        disc_epoch_loss = 0
        batch_cnt = len(dataloader.dataset)

        for step, (img, _) in enumerate(dataloader):
            img = img.to(device)
            
            noise = torch.randn(img.size(0), 100, device = device)

            d_optim.zero_grad()
            real_output = disc(img)
            d_real_loss = loss_function(real_output, torch.ones_like(real_output))
            d_real_loss.backward()

            gen_img = gen(noise)
            fake_output = disc(gen_img.detach())
            d_fake_loss = loss_function(fake_output, torch.zeros_like(fake_output))
            d_fake_loss.backward()

            d_loss = d_real_loss + d_fake_loss
            d_optim.step()

            g_optim.zero_grad()
            fake_output = disc(gen_img)
            g_loss = loss_function(fake_output, torch.ones_like(fake_output))

            g_loss.backward()
            g_optim.step()

            with torch.no_grad():
                disc_epoch_loss += d_loss
                gen_epoch_loss += g_loss

        with torch.no_grad():
            disc_epoch_loss /= batch_cnt
            gen_epoch_loss /= batch_cnt
            D_loss.append(disc_epoch_loss)
            G_loss.append(gen_epoch_loss)
            print(f"{epoch} | Generator_loss : {gen_epoch_loss}, Discriminator_loss : {disc_epoch_loss}")
            epoch += 1
            display(gen, test_input, epoch)

    torch.save(gen.state_dict(), f"./model/gen.pt")
    torch.save(disc.state_dict(), f"./model/disc.pt")
    torch.save(gen.state_dict(), f"./train/{time}/gen.pt")
    torch.save(disc.state_dict(), f"./train/{time}/disc.pt")