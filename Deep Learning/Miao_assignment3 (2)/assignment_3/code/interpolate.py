import argparse
import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets
from torch.autograd import Variable

from a3_gan_template import Generator


#checkpoint = torch.load(args.resume)
#args.latent_dim = 100
genertor = Generator()
genertor.load_state_dict(torch.load('mnist_generator.pt'))
feature = torch.rand(30, 100)

gen_images = genertor(feature)

sampled_image=gen_images[:2]
i_l=torch.zeros(9,784)
for i in range(9):
    i_l[i,:]=(gen_images[1]- gen_images[0])/9*i+gen_images[0]

save_image(i_l.reshape(9,1,28,28),
                            'gan_images/{}.png',
                            nrow=9, normalize=True)
              
#genertor.eval()