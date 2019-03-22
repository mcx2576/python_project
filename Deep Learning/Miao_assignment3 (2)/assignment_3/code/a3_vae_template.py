import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torch.nn import functional as F
from datasets.bmnist import bmnist
import scipy.stats

class Encoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()
        
        self.fc1 = nn.Linear(784, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, z_dim)
        self.fc22 = nn.Linear(hidden_dim, z_dim)
    def forward(self, input):
        """
        Perform forward pass of encoder.
        Returns mean and std with shape [batch_size, z_dim]. Make sure
        that any constraints are enforced.
        """
        hmu =  F.tanh(self.fc1(input))
        hstd =  F.sigmoid(self.fc1(input))

        mean, std =self.fc21(hmu), self.fc22(hstd)
     #   raise NotImplementedError()

        return mean, std


class Decoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()
        self.fc1 = nn.Linear( z_dim, hidden_dim)
        self.fc2 = nn.Linear( hidden_dim, 784)
    def forward(self, input):
        """
        Perform forward pass of encoder.

        Returns mean with shape [batch_size, 784].
        """
    #    print(input.shape)
        hz = F.relu(self.fc1(input))
    
        
        mean = F.sigmoid(self.fc2(hz))
      #  raise NotImplementedError()

        return mean


class VAE(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()

        self.z_dim = z_dim
        self.encoder = Encoder(hidden_dim, z_dim)
        self.decoder = Decoder(hidden_dim, z_dim)

    def forward(self, input):
        """
        Given input, perform an encoding and decoding step and return the
        negative average elbo for the given batch.
        """
      #  x = input.view(-1,784)
        mu, std = self.encoder(input)
        
        self.z = mu + torch.randn(mu.shape)*std
 
        self.x_out =self.decoder(self.z)
        epsilon = 1e-8
        b_loss = -torch.sum((input*torch.log(epsilon +self.x_out)+(1-input)*torch.log(epsilon +1-self.x_out)), dim=1)
       # print(torch.transpose(mu,0,1).shape)
        KL_loss = 0.5*(torch.sum(mu**2, dim=1)-mu.shape[1]-torch.sum(torch.log(epsilon+std**2), dim=1)+torch.sum( (epsilon+std**2),dim=1))
        #print(KL_loss)
        average_negative_elbo = b_loss + KL_loss
       # average_negative_elbo = None
     #   raise NotImplementedError()
        average_negative_elbo = torch.sum(average_negative_elbo)
        average_negative_elbo/=input.shape[0]
        return average_negative_elbo

    def sample(self, n_samples):
        """
        Sample n_samples from the model. Return both the sampled images
        (from bernoulli) and the means for these bernoullis (as these are
        used to plot the data manifold).
        """
        
        #indices =  torch.LongTensor(n_samples).random_(0,self.z.shape[0])
        Z = torch.randn(n_samples, self.z_dim)
        im_means = self.decoder(Z).detach().numpy()
        imgs_array=np.ones([n_samples, 784],dtype=np.int32)
        sampled_ims = torch.from_numpy(np.random.binomial(imgs_array, im_means)).float()

     #   raise NotImplementedError()

        return sampled_ims, im_means


def epoch_iter(model, data, optimizer):
    """
    Perform a single epoch for either the training or validation.
    use model.training to determine if in 'training mode' or not.

    Returns the average elbo for the complete epoch.
    """
    average_negative_elbo = 0
    for data_batch in data:
        batch_size = data_batch.shape[0]
        data_batch = data_batch.reshape(batch_size, 784)
        optimizer.zero_grad()
        loss = model(data_batch)
        loss.backward()
        optimizer.step()
        average_negative_elbo+=loss
    average_epoch_elbo= average_negative_elbo /-len(data)
  #  raise NotImplementedError()

    return average_epoch_elbo


def run_epoch(model, data, optimizer):
    """
    Run a train and validation epoch and return average elbo for each.
    """
    traindata, valdata = data

    model.train()
    train_elbo = epoch_iter(model, traindata, optimizer)

    model.eval()
    val_elbo = epoch_iter(model, valdata, optimizer)

    return train_elbo, val_elbo


def save_elbo_plot(train_curve, val_curve, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(train_curve, label='train elbo')
    plt.plot(val_curve, label='validation elbo')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('ELBO')
    plt.tight_layout()
    plt.savefig(filename)


def main():
    data = bmnist()[:2]  # ignore test split
    model = VAE(z_dim=ARGS.zdim)
    optimizer = torch.optim.Adam(model.parameters())

    train_curve, val_curve = [], []
    for epoch in range(ARGS.epochs):
        if epoch == 0:
            z= torch.randn(25, ARGS.zdim).detach()  
            sampled_ims = model.decoder(z)
            imgs=sampled_ims.reshape(25,1,28,28)
            grid_imgs=make_grid(imgs, nrow=5)
            plt.imshow(grid_imgs.detach().numpy().transpose((1, 2, 0)))
            plt.show()

      #  if (epoch+1) %5==0:
         #   sampled_ims, sample_mean = model.sample(25)
           # print(sampled_ims)
          #  imgs=sampled_ims.reshape(25,1,28,28)
          #  grid_imgs=make_grid(imgs, nrow=5)
           # plt.imshow(grid_imgs.permute(1, 2, 0))
           # plt.show()
        elbos = run_epoch(model, data, optimizer)
        train_elbo, val_elbo = elbos
        train_curve.append(train_elbo)
        val_curve.append(val_elbo)
        print(f"[Epoch {epoch}] train elbo: {train_elbo} val_elbo: {val_elbo}")

        # --------------------------------------------------------------------
        #  Add functionality to plot samples from model during training.
        #  You can use the make_grid functioanlity that is already imported.
        # --------------------------------------------------------------------
        

    # --------------------------------------------------------------------
    #  Add functionality to plot plot the learned data manifold after
    #  if required (i.e., if zdim == 2). You can use the make_grid
    #  functionality that is already imported.
    # --------------------------------------------------------------------
    
    if ARGS.zdim==2:
        num_point=20
        p = np.linspace(0,1,num_point+2)[1:-1]
        z = scipy.stats.norm.ppf(p)
        z_l = []
        for i in range(num_point):
            for j in range(num_point):
                z_l.append([z[i],z[j]])
        z_l = torch.from_numpy(np.array(z_l)).float()

        im_mean = model.decoder(z_l)
        imgs=im_mean.reshape(400,1,28,28)
        grid_imgs=make_grid(imgs, nrow=20)
        plt.imshow(grid_imgs.detach().numpy().transpose(1, 2, 0))
        plt.show()
        
    save_elbo_plot(train_curve, val_curve, 'elbo.pdf')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=40, type=int,
                        help='max number of epochs')
    parser.add_argument('--zdim', default=20, type=int,
                        help='dimensionality of latent space')

    ARGS = parser.parse_args()

    main()
