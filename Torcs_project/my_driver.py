from pytocl.driver import Driver
from pytocl.car import State, Command


###
import numpy as np

# help functions if needed
def to_0_1(x):
    if x < 0.5:
        return 0.0
    else:
        return 1.0

def put_in_range(a,b,x):
    if x < a:
        return a
    elif x > b:
        return b
    else:
        return x

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



class MyDriver(Driver):


    def __init__(self):
        super(MyDriver, self).__init__()
        self.qqq=0
        self.prev_gear = -1

        self.rev = 0
        self.rev_t = -1
        self.num_stand_still = 0

        trained_torcs_NN_filename_acc = "pytocl/torcs_NN_trained_acc_new_q_150"
        trained_torcs_NN_filename_brake = "pytocl/torcs_NN_trained_brake_new_q_150"
        trained_torcs_NN_filename_steer = "pytocl/torcs_NN_trained_steer_new_q_150"
        num_features = 22
        num_nodes_hidden = 100
        self.trained_torcs_NN_acc = torcs_mlp_NN_acc(3, 100, 1)
        self.trained_torcs_NN_acc.load_state_dict(torch.load(trained_torcs_NN_filename_acc))
        self.trained_torcs_NN_brake = torcs_mlp_NN_brake(3, 150, 1)
        self.trained_torcs_NN_brake.load_state_dict(torch.load(trained_torcs_NN_filename_brake))
        self.trained_torcs_NN_steer = torcs_mlp_NN_steer(22, 150, 1)
        self.trained_torcs_NN_steer.load_state_dict(torch.load(trained_torcs_NN_filename_steer))


    def drive(self, carstate: State) -> Command:
        """
        Produces driving command in response to newly received car state.

        This is a dummy driving routine, very dumb and not really considering a
        lot of inputs. But it will get the car (if not disturbed by other
        drivers) successfully driven along the race track.
        """
        self.qqq =self.qqq+1

        if abs(carstate.speed_x) <= 2 and self.qqq > 300:
            print("stand still" + str(self.num_stand_still))
            self.num_stand_still = self.num_stand_still + 1
        else:
            self.num_stand_still = 0

        if self.num_stand_still >= 30:
            # reverse 40 timesteps
            self.rev = 1
            self.rev_t = 40

        if self.rev == 1:
            # reversing
            print("rev now, rev_t" + str(self.rev_t) + "!")
            if self.rev_t == 0:
                # finished reversing, go back to not reversing
                self.rev = 0
                self.qqq=0
                command = Command() 
                command.accelerator = 1.0
                command.brake = 0.0
                command.gear = 1
                command.steering=0
                return command
            else:
                # continue reversing
                self.rev_t = self.rev_t - 1
                command = Command()
                command.accelerator = 1.0
                command.brake = 0.0
                command.gear = -1
                self.steer(carstate, 0.0, command)
                return command

        # out of track, go back to track
        if carstate.distances_from_egde_valid == False:
            command = Command()
            command.gear = 1
            self.accelerate(carstate,60,command)
            self.steer(carstate, 0.0, command)
            return command

        # input for steering
        test_x = np.zeros((1,22),dtype=np.float32)
        test_x[0][0] = carstate.speed_x
        test_x[0][1] = carstate.distance_from_center
        test_x[0][2] = carstate.angle
        test_x[0][3:] = np.array(carstate.distances_from_edge)
        test_x = Variable(torch.Tensor(test_x))

        # input for brake
        new_x = np.zeros((1,3),dtype=np.float32)
        new_x[0][0] = self.prev_gear
        new_x[0][1] = carstate.speed_x
        new_x[0][2] = carstate.distances_from_edge[9]
        new_x = Variable(torch.Tensor(new_x))

        q_x = np.zeros((1,3),dtype=np.float32)
        q_x[0][0] = self.prev_gear
        q_x[0][1] = carstate.speed_x
        q_x[0][2] = carstate.distances_from_edge[9]
        q_x = Variable(torch.Tensor(q_x))

        pred_y_acc = self.trained_torcs_NN_acc(q_x)
        pred_y_brake = self.trained_torcs_NN_brake(new_x)
        pred_y_steer = self.trained_torcs_NN_steer(test_x)

        # create command
        command = Command()
        command.accelerator = put_in_range(0.0,1.0,pred_y_acc.data[0][0])
        command.brake = put_in_range(0.0,1.0, pred_y_brake.data[0][0])

        if command.accelerator > 0:
            if carstate.rpm > 8000:
                command.gear = carstate.gear + 1
        # else:
        #     command.brake = min(-acceleration, 1)
        if carstate.rpm < 2500:
            command.gear = carstate.gear - 1

        if not command.gear:
            command.gear = carstate.gear or 1

        command.steering = put_in_range(-1.0,1.0,pred_y_steer.data[0][0])
        command.focus = 0.0

        if self.data_logger:
            self.data_logger.log(carstate, command)

        # remember gear
        self.prev_gear = command.gear

        return command


