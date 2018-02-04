import numpy as np
import math
import csv

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

###################################
# NN constants
NUM_NODES_HIDDEN = 20

class torcs_mlp_NN_acc(nn.Module):

    def __init__(self, num_features, num_nodes_hidden, num_output):
        super(torcs_mlp_NN_acc, self).__init__()

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

###################################
# Read from training data file
training_data_filenames = ['train_data_EA.csv']#['data_track1.csv', 'data_track2.csv']
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
            data_y.append([float(row[1])])
            data_x.append([float(s) for s in row[4:26]])
#            data_x.append([float(row[5]),float(row[6]),float(row[7]),float(row[8]),float(row[17])])
#            data_x.append([float(row[5]),float(row[6]),float(row[7]),float(row[8]),float(row[11]), \
#                           float(row[14]),float(row[17]),float(row[20]),float(row[23])])

num_data_points = len(data_y)
y = torch.Tensor(data_y)
data_x = torch.Tensor(data_x)


total_input_vars = 22#9

###################################
# Functions

def calc_fitness( gene_indiv, performance ):
    #so after this we have: rank[0] gives index of worst gene, rank[POP_SIZE-1] of best
    rank = np.argsort(performance)

    fitness = np.zeros(POP_SIZE)
    for i in range(0,POP_SIZE):
#        fitness[rank[i]] = 10*i
#        print(str(rank[i])+","+str(i)+","+str(POP_SIZE*i - POP_SIZE*sum(gene_indiv[rank[i]])))
#        fitness[rank[i]] = (max(0,POP_SIZE*i - POP_SIZE*sum(gene_indiv[rank[i]])))**1.3
        fitness[rank[i]] = ( (1- (sum(gene_indiv[rank[i]])-1)/total_input_vars ) *i )**3

    return fitness

# Assumes data is global
def err_NN( gene ):

    # Create NN corresponding with gene
    num_input = int(sum(gene))
    if num_input == 0:
        return -1

    print("current gene: " + str(gene))
    indiv_NN = torcs_mlp_NN_acc(num_input, NUM_NODES_HIDDEN, 1)

    # Training method (Chosen as Adam, MSELOSS)
    optimizer = torch.optim.Adam(indiv_NN.parameters())
    loss_func = torch.nn.MSELoss()

    num_iter = 4
    train_errors = np.zeros(num_iter)
    for i in range(0,num_iter):
        time_start = time.time()

        idx = np.random.permutation(num_data_points)
        for n in range(0, num_data_points):
            # Select which k out of num_input data-x's to use
            input_x = np.zeros(num_input,dtype=np.float32)
            q=0
            for k in range(0,total_input_vars):
                if gene[k] == 1:
                    input_x[q] = data_x[idx[n]][k]
                    q=q+1
            input_x = Variable(torch.Tensor(input_x))
            target_y = Variable(torch.Tensor(y[idx[n]]))

            #
            output_y = indiv_NN(input_x)
            loss = loss_func(output_y, target_y)
            train_errors[i] = train_errors[i] + loss.data[0]#np.log(loss.data[0])

            #
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

        print("iteration: %r, avg train err: %.10f, time: %.2f sec" %
                (i, train_errors[i]/num_data_points, time.time()-time_start) )

    # final train err
    final_train_err =[]
    err=0.0
    for n in range(0,num_data_points):
        input_x = np.zeros(num_input,dtype=np.float32)
        q=0
        for k in range(0,total_input_vars):
            if gene[k] == 1:
                input_x[q] = data_x[n][k]
                q=q+1
        input_x = Variable(torch.Tensor(input_x))
        target_y = Variable(torch.Tensor(y[n]))
        output_y = indiv_NN(input_x)
        loss = loss_func(output_y, target_y)
        final_train_err.append(loss.data[0])
        #print(loss.data[0])
        err = err + loss.data[0]

    print("final avg train err: %.10f, time: %.2f sec, 1/(1+train error): %.6f" %
           (err/num_data_points, time.time()-time_start, 1.0/(1.0+(err/num_data_points))) )
    avg_fin_err = err / num_data_points
    print("sum (final_train_err[i] - avg)**2: %.2f" %
        sum( [(final_train_err[i]-avg_fin_err)**2 for i in range(0,num_data_points)] ) )

    # Final error
    return 1.0/(1+sum( [(final_train_err[i]-avg_fin_err)**2 for i in range(0,num_data_points)] ))

# NOTE: Assumes fitness array contains only vals >= 0
def select_roulette_wheel( fitness ):
    num_individuals = len(fitness) #len=POP_SIZE
    cumul_pieces_width = np.zeros(num_individuals)
    for i in range(0,num_individuals):
        cumul_pieces_width[i] = sum(fitness[0:i+1])

    total_width = cumul_pieces_width[-1]

    r = total_width*np.random.rand()
    for i in range(0,num_individuals):
        if r < cumul_pieces_width[i]:
            return i

# 1-point crossover (tail-swap)
def perform_crossover( gene_1, gene_2 ):
    gene_length = len(gene_1)
    intersection_point = 1+np.random.randint(gene_length-1)

    gene_1_copy = gene_1.copy()
    for k in range(intersection_point,gene_length):
        gene_1[k] = gene_2[k]
        gene_2[k] = gene_1_copy[k]
# gene = 0 1 0 1, g_len = 4, inter_p = 1+rand(0,1,2)= 1 / 2 / 3

# bitflip mutation
def perform_mutation( gene, pos ):
    if gene[pos] == 1:
        gene[pos] = 0
    else:
        gene[pos] = 1

###################################

# Constants
POP_SIZE = 10
P_CROSSOVER = 0.8
P_MUTATION = 0.1
NUM_EA_iter = 30

# Initialization
gene_indiv = np.zeros((POP_SIZE, total_input_vars))
for i in range(0,POP_SIZE):
    # Initialize gene randomly (uniformly choose 0 or 1 per coordinate)
    for k in range(0,total_input_vars):
        gene_indiv[i][k] = np.random.randint(2)

#gene_indiv[0] = np.array([1.0,0.0,0.0,0.0,0.0,0.0])


gene_avgs = np.zeros(total_input_vars)

for q in range(0,NUM_EA_iter):

    # Selection
    # 1. Calculate fitness (>0 to maximize) of all individuals
    #fitness = np.zeros(POP_SIZE)
    performance = np.zeros(POP_SIZE)
    for i in range(0,POP_SIZE):
        performance[i] = err_NN( gene_indiv[i] )
#        #fitness[i] = -1.0*err_NN( gene_indiv[i] )
#        fitness[i] = calc_fitness( gene_indiv[i] )
    fitness = calc_fitness( gene_indiv, performance )


    # 2. Now we know fitness of all individuals,
    # perform "roulette wheel" selection technique.
    # (is selection with replacement technique)
    selected = np.zeros(POP_SIZE)
    for i in range(0,POP_SIZE):
        selected[i] = select_roulette_wheel( fitness )

    for m in range(0,total_input_vars):
        gene_avgs[m] = gene_avgs[m] + sum(gene_indiv[:,m])

    print("iteration: " + str(q))
    print(gene_indiv)
    print(fitness)
    print(selected)
    if q == (NUM_EA_iter-1):
        print("finish:")
        print(gene_indiv)
        print(fitness)
        print(selected)
        gene_avgs = 1.0/(POP_SIZE*NUM_EA_iter) *np.array(gene_avgs)
        print("gene avgs is: " + str(gene_avgs))
        break


    # So now we have selected POP_SIZE individuals of population (call them parents)
    pop_genes_copy = gene_indiv.copy()
    #print(pop_genes_copy)
    for i in range(0,10):
        gene_indiv[i] = pop_genes_copy[int(selected[i])]
    #print(gene_indiv)

    # Now replace all parents with children:
    # 1. Perform crossover
    for m in range(0,int(POP_SIZE/2)):
        if (np.random.rand() < P_CROSSOVER) == True:
            #print("crossover")
            #print(gene_indiv[2*m], gene_indiv[2*m+1])
            perform_crossover( gene_indiv[2*m], gene_indiv[2*m+1] )
            #print(gene_indiv[2*m], gene_indiv[2*m+1])

    # 2. Perform mutation
    for i in range(0,POP_SIZE):
        #print("mutation")
        #print(gene_indiv[i])
        for k in range(0,total_input_vars):
            if (np.random.rand() < P_MUTATION):
                perform_mutation( gene_indiv[i], k )
        #print(gene_indiv[i])
    # Check termination:


#####################################################



