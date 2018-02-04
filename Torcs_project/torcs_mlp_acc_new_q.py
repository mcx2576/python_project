import csv
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class torcs_mlp_NN_acc(nn.Module):

    def __init__(self, num_features, num_nodes_hidden, num_output):
        super(torcs_mlp_NN_acc, self).__init__()

        # 3 hidden layer
        self.lin1 = nn.Linear(num_features, num_nodes_hidden)
        self.lin2 = nn.Linear(num_nodes_hidden, num_nodes_hidden)
        self.lin3 = nn.Linear(num_nodes_hidden, num_nodes_hidden)
        self.lin4 = nn.Linear(num_nodes_hidden, num_output)

    def forward(self, x):
        y = F.relu(self.lin1(x))
        y = F.relu(self.lin2(y))
        y = F.relu(self.lin3(y))
        y = F.sigmoid(self.lin4(y))
        return y


# define the NN
num_features = 3
num_nodes_hidden = 100
num_output = 1
torcs_NN_acc = torcs_mlp_NN_acc(num_features, num_nodes_hidden, num_output)

print(torcs_NN_acc)


# training stage

print("start training")

#learning_rate = 0.05
#optimizer = torch.optim.SGD(my_torcs_NN.parameters(), lr=learning_rate)
loss_func = torch.nn.MSELoss()


# read in training data file
training_data_filenames = ['train_data1.csv','train_data2.csv','train_data3.csv','train_data4.csv','train_data5.csv','train_data6.csv','train_data7.csv','train_data8.csv']#['data_track1.csv', 'data_track2.csv']
num_training_data_files = len(training_data_filenames)
data_y = []
data_x = []
for n in range(0, num_training_data_files):
    with open(training_data_filenames[n], 'r') as fp:
        reader = csv.reader(fp, delimiter=',')
        col_headers = next(reader, None)
        for row in reader:
#       0    1     2    3    4    5      6     7      8    ...   17    ...   26
#DATA: acc brake steer gear rpm speed dist_c angle edge[0] ... edge[9] ... edge[18]
            data_y.append([float(row[0])])
            data_x.append([float(row[3]), float(row[5]), float(row[17])])
#            data_x.append([float(s) for s in row[3:6]])

num_data_points = len(data_y)
data_y = torch.Tensor(data_y)
data_x = torch.Tensor(data_x)


# we do NOT normalize input anymore


# read in validation data
val_data_filenames = ['val_data1.csv','val_data2.csv','train_data3.csv','train_data4.csv','train_data5.csv','train_data6.csv']#['data_track1.csv', 'data_track2.csv']
num_val_data_files = len(val_data_filenames)
val_data_y = []
val_data_x = []
for n in range(0, num_val_data_files):
    with open(val_data_filenames[n], 'r') as fp:
        reader = csv.reader(fp, delimiter=',')
        col_headers = next(reader, None)
        for row in reader:
#       0    1     2    3    4    5      6     7      8    ...   17    ...   26
#DATA: acc brake steer gear rpm speed dist_c angle edge[0] ... edge[9] ... edge[17]
            val_data_y.append([float(row[0])])
            val_data_x.append([float(row[3]), float(row[5]), float(row[17])])
#            val_data_x.append([float(s) for s in row[3:6]])

num_val_data_points = len(val_data_y)
val_data_y = torch.Tensor(val_data_y)
val_data_x = torch.Tensor(val_data_x)



####################################333

num_iter = 15
for i in range(0, num_iter):

    # diff learning rate
    if i <= 1:
        learning_rate = 0.05
    elif i <= 5:
        learning_rate = 0.02
    elif i <= 10:
        learning_rate = 0.01
    else:
        learning_rate = 0.005
#    optimizer = torch.optim.SGD(torcs_NN_acc.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adam(torcs_NN_acc.parameters())

    # remember how long 1 iteration takes
    time_start = time.time()
    train_err = 0.0

    k=0
    idx = np.random.permutation(num_data_points)
    for n in range(0, num_data_points):
        if k % 4000 == 0:
            print(k)
        k=k+1

        # convert data to use for NN
        input_x = Variable(data_x[idx[n]])
        target_y = Variable(data_y[idx[n]])

        # Forward propagation, and error calculation
        output_y = torcs_NN_acc(input_x)
        loss = loss_func(output_y, target_y)
        train_err = train_err + loss.data[0]

        # Reset optimizer, and backward propagation
        optimizer.zero_grad()
        loss.backward()

        # Update weights of NN
        optimizer.step()

    print("iteration: %r, avg train err: %.10f, time: %.2f sec" %
            (i, train_err/num_data_points, time.time()-time_start) )


    # check error on validation set
    time_start = time.time()
    val_err = 0.0
    for n in range(0, num_val_data_points):
        input_x = Variable(val_data_x[n])
        target_y = Variable(val_data_y[n])

        output_y = torcs_NN_acc(input_x)
        loss = loss_func(output_y, target_y)
        val_err = val_err + loss.data[0]

    print("iteration: %r, avg val err: %.10f, time: %.2f sec" %
            (i, val_err/num_val_data_points, time.time()-time_start) )


#####################################################


# save and load a neural network (only parameters)
torch.save(torcs_NN_acc.state_dict(), "torcs_NN_trained_acc_new_q")

#
torcs_NN_acc2 = torcs_mlp_NN_acc(num_features, num_nodes_hidden, num_output)
torcs_NN_acc2.load_state_dict(torch.load("torcs_NN_trained_acc_new_q"))



