from pytocl.driver import Driver
from pytocl.car import State, Command
import numpy as np
from random import randint
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import subprocess
import threading
import queue

def torchmodel_to_numpy(model):    
    #model FILENAME
    l1_weight = model['lin1.weight'].numpy()
    l2_weight = model['lin2.weight'].numpy()
    l3_weight = model['lin3.weight'].numpy()
    l4_weight = model['lin4.weight'].numpy()

    l1_bias = model['lin1.bias'].numpy()
    l2_bias = model['lin2.bias'].numpy()
    l3_bias = model['lin3.bias'].numpy()
    l4_bias = model['lin4.bias'].numpy()
    
    weights = [l1_weight, l2_weight, l3_weight, l4_weight]
    biases = [l1_bias, l2_bias, l3_bias, l4_bias]
           
    return weights, biases
    
def numpy_to_torchmodel(model, weights, biases):
    
    # print(trained_torcs_NN_acc.lin1.weight)
    # trained_torcs_NN_acc.lin1.weight = torch.nn.Parameter(torch.FloatTensor(20,1))
    
    model.lin1.weight = torch.nn.Parameter(torch.from_numpy(weights[0]))
    model.lin2.weight = torch.nn.Parameter(torch.from_numpy(weights[1]))
    model.lin3.weight = torch.nn.Parameter(torch.from_numpy(weights[2]))
    model.lin4.weight = torch.nn.Parameter(torch.from_numpy(weights[3]))
    
    
    model.lin1.bias = torch.nn.Parameter(torch.from_numpy(biases[0]))
    model.lin2.bias = torch.nn.Parameter(torch.from_numpy(biases[1]))
    model.lin3.bias = torch.nn.Parameter(torch.from_numpy(biases[2]))
    model.lin4.bias = torch.nn.Parameter(torch.from_numpy(biases[3]))    
    
    return model

def convert_params_to_dict(acc_w, acc_b, brake_w, brake_b, steer_w, steer_b):
    indv_dict = {'acc_w':acc_w, 'acc_b':acc_b, 'brake_w': brake_w,'brake_b': brake_b, 'steer_w': steer_w, 'steer_b': steer_b }
    return indv_dict

def run_server(race):
    
    command = "torcs -r /home/ece/Desktop/ciproject/torcs-server/" + race +".xml"
    proc = subprocess.Popen([command],shell = True,stdout = subprocess.PIPE).communicate()
    #print(proc)
    
def run_client(que):
    proc = subprocess.Popen(["./start.sh"],shell = True,stdout = subprocess.PIPE).communicate()
    results = proc[0].decode("utf-8").split('\n')[:-1]
    
    que.put(results)

def run(race):
    try:
        print(race)
        print("starting client")
        que = queue.Queue()
        t1 = threading.Thread(target=run_client, args = [que])
        t1.start()

        print("starting server")
        t2 = threading.Thread(target=run_server, args = (race,))
        t2.start()
        t2.join()

        print("server joined")
        
        t1.join()
        print("client joined")
            
        res = que.get()
       
        return(res)
            
    except:
        print("Error: threads not started")
    

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

class torcs_mlp_NN_brake(nn.Module):

    def __init__(self, num_features, num_nodes_hidden, num_output):
        super(torcs_mlp_NN_brake, self).__init__()

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

class torcs_mlp_NN_steer(nn.Module):

    def __init__(self, num_features, num_nodes_hidden, num_output):
        super(torcs_mlp_NN_steer, self).__init__()

        # 3 hidden layer
        self.lin1 = nn.Linear(num_features, num_nodes_hidden)
        self.lin2 = nn.Linear(num_nodes_hidden, num_nodes_hidden)
        self.lin3 = nn.Linear(num_nodes_hidden, num_nodes_hidden)
        self.lin4 = nn.Linear(num_nodes_hidden, num_output)

    def forward(self, x):
        y = F.relu(self.lin1(x))
        y = F.relu(self.lin2(y))
        y = F.relu(self.lin3(y))
        y = F.tanh(self.lin4(y))
        return y

def race_all(races):
    distpos_races = []
    
    for track in races:
        distpos = [] #comprised of tuples (distance_from_start, race_position)
        results = run(track)

        for r in results:
            d, p = r.split(" ")
            distpos.append((float(d), int(p)))    

        distpos_races.append(distpos)

    return distpos_races

generations = 100
crossover_prob = 0.7
mutation_prob = 0.05
mean = 0
sd = 0.01
races = ["forza","aalborg","alpine","fspeedway","dirt1","dirt2"]
c = 0.9
success_rule = 0.2 #1/5 rule
k = 5 #check success every 5 generation
ps = 0 #success rate as probability


def read_network():
    
    trained_torcs_NN_filename_acc = "pytocl/torcs_NN_trained_acc_new"
    trained_torcs_NN_filename_brake = "pytocl/torcs_NN_trained_brake_new"
    trained_torcs_NN_filename_steer = "pytocl/torcs_NN_trained_steer_new_good_22_input"

    trained_torcs_NN_acc = torcs_mlp_NN_acc(1, 20, 1)
    trained_torcs_NN_acc.load_state_dict(torch.load(trained_torcs_NN_filename_acc))
    trained_torcs_NN_brake = torcs_mlp_NN_brake(3, 20, 1)
    trained_torcs_NN_brake.load_state_dict(torch.load(trained_torcs_NN_filename_brake))
    trained_torcs_NN_steer = torcs_mlp_NN_steer(22, 20, 1)
    trained_torcs_NN_steer.load_state_dict(torch.load(trained_torcs_NN_filename_steer))

    acc_w, acc_b = torchmodel_to_numpy(torch.load(trained_torcs_NN_filename_acc))
    brake_w, brake_b = torchmodel_to_numpy(torch.load(trained_torcs_NN_filename_brake))
    steer_w, steer_b = torchmodel_to_numpy(torch.load(trained_torcs_NN_filename_steer))

    return convert_params_to_dict(acc_w, acc_b, brake_w, brake_b, steer_w, steer_b),trained_torcs_NN_acc,trained_torcs_NN_brake,trained_torcs_NN_steer

def write_network(trained_torcs_NN_acc, acc_w, acc_b, trained_torcs_NN_brake, brake_w, brake_b,trained_torcs_NN_steer, steer_w, steer_b):
    
    torcs_NN_acc = numpy_to_torchmodel(trained_torcs_NN_acc, acc_w, acc_b)
    torcs_NN_brake = numpy_to_torchmodel(trained_torcs_NN_brake, brake_w, brake_b)
    torcs_NN_steer = numpy_to_torchmodel(trained_torcs_NN_steer, steer_w, steer_b)
    
    torch.save(torcs_NN_acc.state_dict(), "pytocl/torcs_NN_trained_acc_new")
    torch.save(torcs_NN_brake.state_dict(), "pytocl/torcs_NN_trained_brake_new")
    torch.save(torcs_NN_steer.state_dict(), "pytocl/torcs_NN_trained_steer_new_good_22_input")

def crossover(indv1, indv2, prob):
    print('crossover')
    
    for k in indv1:
        for i in range(len(indv1[k])):
            if randint(1, 100) < prob*100:
                #perform crossover
                l = len(indv1[k][i])
                split = randint(0,l)
                x1 = indv1[k][i][0:split]
                x2 = indv2[k][i][split:l]
                
                temp = []
                temp.extend(x1)
                temp.extend(x2)
                
                indv1[k][i] = np.asarray(temp, dtype=np.float32)
                
                y1 = indv2[k][i][0:split]
                y2 = indv1[k][i][split:l]
                
                temp = []
                temp.extend(y1)
                temp.extend(y2)
                indv2[k][i] = np.asarray(temp, dtype=np.float32)
                               
    return indv1, indv2

def mutate(indv, prob, mean, sd):
    print('mutate')
    for key, value in sorted(indv.items()):
        for l in value:
            for i in range(len(l)):
                if randint(1, 100) < prob*100:
                    #perform mutation
                    
                    l[i] = l[i] + np.random.normal(mean, sd)

    return indv
  
def breed(indv, fitness, crossover_prob, mutation_prob, mean, sd):
    
    indv_updated = mutate(indv, mutation_prob,mean,sd)
#     indv1,indv2 = crossover(indv0,indv0, crossover_prob)

    return indv_updated

def get_fitness(distance_position):
    
    fitness = 0
    
    sum_positions = 0
    count = 0    
    final_positions = 0
    
    for dps in distance_position:
        for dp in dps:
            sum_positions = sum_positions + dp[1]
            count = count + 1
            
    mean_pos = sum_positions / count
    
    for dps in distance_position:
        final_positions = final_positions + dps[-1][1]
    
    final_pos = final_positions / len(distance_position)
    
    fitness = (mean_pos + final_pos) / 2.0
    
    return fitness


fit = -1

#two-membered evolutionary strategies utilizing mutation of weights and adaptation of mutation step size
for g in range(generations):
    
    print(fit)
    if fit == -1: 
        #first run
        dp = race_all(races)

        fit = get_fitness(dp)

        indv,trained_torcs_NN_acc,trained_torcs_NN_brake,trained_torcs_NN_steer = read_network()

        indv_updated = breed(indv, fit, crossover_prob, mutation_prob, mean, sd)

        write_network(trained_torcs_NN_acc, indv_updated['acc_w'],indv_updated['acc_b'], trained_torcs_NN_brake, indv_updated['brake_w'],indv_updated['brake_b'],trained_torcs_NN_steer, indv_updated['steer_w'],indv_updated['steer_b'])

        dp2 = race_all(races)

        fit2 = get_fitness(dp2)

        if(fit > fit2):
            #keep the previous network
            write_network(trained_torcs_NN_acc, indv['acc_w'],indv['acc_b'], trained_torcs_NN_brake, indv['brake_w'],indv['brake_b'],trained_torcs_NN_steer, indv['steer_w'],indv['steer_b'])
        else:
            fit = fit2
            ps = ps + 1 #successful mutation
            
    else:
        indv,trained_torcs_NN_acc,trained_torcs_NN_brake,trained_torcs_NN_steer = read_network()

        indv_updated = breed(indv, fit, crossover_prob, mutation_prob, mean, sd)

        write_network(trained_torcs_NN_acc, indv_updated['acc_w'],indv_updated['acc_b'], trained_torcs_NN_brake, indv_updated['brake_w'],indv_updated['brake_b'],trained_torcs_NN_steer, indv_updated['steer_w'],indv_updated['steer_b'])

        dp2 = race_all(races)

        fit2 = get_fitness(dp2)

        if(fit < fit2):
            #keep the previous network
            write_network(trained_torcs_NN_acc, indv['acc_w'],indv['acc_b'], trained_torcs_NN_brake, indv['brake_w'],indv['brake_b'],trained_torcs_NN_steer, indv['steer_w'],indv['steer_b'])
        else:
            fit = fit2
            
            ps = ps + 1 #successful mutation
        
    if (g+1)%5 == 0:
        #check 1/5 rule
        ps = ps / 5
        
        #uncorrelated sd update
        if ps > 0.2:
            sd = sd / c
        elif ps < 0.2:
            sd = sd*c
        else:
            sd = sd
        ps = 0
        
        #boundary rule
        if sd < 0.01:
            sd = 0.01
