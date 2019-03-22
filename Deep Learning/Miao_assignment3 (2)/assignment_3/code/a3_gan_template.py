import argparse
import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets
from torch.autograd import Variable

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # Construct generator. You are free to experiment with your model,
        # but the following is a good start:
        #   Linear args.latent_dim -> 128
        #   LeakyReLU(0.2)
        #   Linear 128 -> 256
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 256 -> 512
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 512 -> 1024
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 1024 -> 768
        #   Output non-linearity
        self.model = nn.Sequential(
            nn.Linear(100,128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 784),
            nn.Tanh()
        )
 
    def forward(self, z):
        # Generate images from z
        image = self.model(z)
        
        return image


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # Construct distriminator. You are free to experiment with your model,
        # but the following is a good start:
        #   Linear 784 -> 512
        #   LeakyReLU(0.2)
        #   Linear 512 -> 256
        #   LeakyReLU(0.2)
        #   Linear 256 -> 1
        #   Output non-linearity
        self.model = nn.Sequential(
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    def forward(self, img):
        # return discriminator score for img
        result = self.model(img)
        return result


def train(dataloader, discriminator, generator, optimizer_G, optimizer_D):
    for epoch in range(args.n_epochs):
        for i, (imgs, _) in enumerate(dataloader):

           # imgs.cuda()
            fake_label =  Variable(torch.Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)
            real_label =  Variable(torch.Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
            real_images = imgs.reshape(imgs.shape[0], imgs.shape[1]*imgs.shape[2]*imgs.shape[3])
           # print(real_images.shape)


            # Train Generator
            # ---------------
            optimizer_G.zero_grad()
            z =torch.randn((imgs.shape[0], args.latent_dim), requires_grad=True)
            gen_images = generator(z)
            
            loss = torch.nn.BCELoss()
            generator_loss = loss(discriminator(gen_images), real_label)
            generator_loss.backward(retain_graph=True)
            optimizer_G.step()
            # Train Discriminator
            # -------------------
            optimizer_D.zero_grad()
            loss_1 = loss(discriminator(real_images), real_label)
            loss_2 = loss(discriminator(gen_images), fake_label)
            total_loss = (loss_1+loss_2)/2
            total_loss.backward()
            optimizer_D.step()
            # Save Images
            # -----------

            batches_done = epoch * len(dataloader) + i
            if batches_done % args.save_interval == 0:
                print(batches_done)
                # You can use the function save_image(Tensor (shape Bx1x28x28),
                # filename, number of rows, normalize) to save the generated
                # images, e.g.:

                save_image(gen_images[:25].reshape(25,1,28,28),
                            'images/{}.png'.format(batches_done),
                            nrow=5, normalize=True)
              


def main():
    # Create output image directory
    os.makedirs('images', exist_ok=True)

    # load data
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))])),
        batch_size=args.batch_size, shuffle=True)

    # Initialize models and optimizers
    generator = Generator()
    discriminator = Discriminator()
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr)

    # Start training
    train(dataloader, discriminator, generator, optimizer_G, optimizer_D)

    # You can save your generator here to re-use it to generate images for your
    # report, e.g.:
    torch.save(generator.state_dict(), "mnist_generator.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=200,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='learning rate')
    parser.add_argument('--latent_dim', type=int, default=100,
                        help='dimensionality of the latent space')
    parser.add_argument('--save_interval', type=int, default=500,
                        help='save every SAVE_INTERVAL iterations')
    args = parser.parse_args()

    main()
