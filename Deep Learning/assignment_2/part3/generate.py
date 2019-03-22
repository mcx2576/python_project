from model import TextGenerationModel
from dataset import TextDataset
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

dataset = TextDataset("assets/book_EN_grimms_fairy_tails.txt", 30)
model = TextGenerationModel( 64, 40, dataset._vocab_size, 128, 2)
model.load_state_dict(torch.load('./model.pth'))
sentence=[]
a =  np.random.randint(87, size=1)
first = torch.eye(87)[a].unsqueeze_(0)
sentence.append(dataset._ix_to_char[int(a)])
h=torch.zeros(model.n_l,1,model.f)
c=torch.zeros(model.n_l,1,model.f)
for t in range(30):
                #print(t)
    output, (h, c) = model.rnn(first, (h,c))
    out = model.linear(output)
                # Deal with the temperature paramenter
   # if config.temperature ==1: #or step ==0:
    #     _, b= torch.max(out, 2)
    #else: 
    predicts = torch.exp(out/2)
    x = torch.distributions.Categorical(predicts)
                    
    b = x.sample()
                #    print(b)
                #    _, b=torch.max(out, 2)
    sentence.append(dataset._ix_to_char[int(b)])
    first = torch.eye(dataset._vocab_size)[b]
print('GS: '+str(''.join(str(i) for i in sentence)))


 