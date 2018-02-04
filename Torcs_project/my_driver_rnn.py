from pytocl.driver import Driver
from pytocl.car import State, Command


###
import numpy as np

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

class RNN(nn.Module):

    def __init__(self, num_features, num_hidden_nodes, num_output):
        super(RNN, self).__init__()

        self.rnn = nn.RNN(
            input_size = num_features,
            hidden_size = num_hidden_nodes,
            num_layers = 1,
            batch_first = True,
        )
        self.lin = nn.Linear(num_hidden_nodes, num_output)

    def forward(self, x, h_state):
        r_out, h_state = self.rnn(x, h_state)
        outs = []
        for time_step in range(r_out.size(1)):
            outs.append(self.lin(r_out[:, time_step, :]))
        return torch.stack(outs, dim=1), h_state




class MyDriver(Driver):
    # Override the `drive` method to create your own driver
    ...
    # def drive(self, carstate: State) -> Command:
    #     # Interesting stuff
    #     command = Command(...)
    #     return command

    def __init__(self):
        super(MyDriver, self).__init__()
        self.qqq=0

        # For RNN
        self.h_state_a = None
        self.h_state_b = None

        trained_torcs_NN_filename_a = "pytocl/torcs_RNN_trained_a"
        trained_torcs_NN_filename_b = "pytocl/torcs_RNN_trained_b"
        num_features = 22
        num_nodes_hidden = 30
        num_output_a = 2
        num_output_b = 1
        self.trained_torcs_NN_a = RNN(num_features, num_nodes_hidden, num_output_a)
        self.trained_torcs_NN_a.load_state_dict(torch.load(trained_torcs_NN_filename_a))
        self.trained_torcs_NN_b = RNN(num_features, num_nodes_hidden, num_output_b)
        self.trained_torcs_NN_b.load_state_dict(torch.load(trained_torcs_NN_filename_b))

    def drive(self, carstate: State) -> Command:
        """
        Produces driving command in response to newly received car state.

        This is a dummy driving routine, very dumb and not really considering a
        lot of inputs. But it will get the car (if not disturbed by other
        drivers) successfully driven along the race track.
        """

        print(" " + str(self.qqq))
        print("------------")
        print("speed_x: " + str(carstate.speed_x))
        print("distance_from_center: " + str(carstate.distance_from_center))
        print("angle: " + str(carstate.angle))
        print("distances_from_edge: ")
        print(carstate.distances_from_edge)
        print("------------")
        print(carstate.distances_from_egde_valid)
        if carstate.distances_from_egde_valid == False:
            print("false distanced_from_edge")

        test_x = np.zeros(22, dtype=np.float32)
        test_x[0] = carstate.speed_x
        test_x[1] = carstate.distance_from_center
        test_x[2] = carstate.angle
        test_x[3:] = np.array(carstate.distances_from_edge)


        #for q in range(0,22):
        #    test_x[q] = (test_x[q] - avgs[q])/ranges[q]

        test_x = test_x.reshape(1,1,22)
        test_x = Variable(torch.from_numpy(test_x))

        pred_y_a, self.h_state_a = self.trained_torcs_NN_a(test_x, self.h_state_a)
        self.h_state_a = Variable(self.h_state_a.data)
        pred_y_b, self.h_state_b = self.trained_torcs_NN_b(test_x, self.h_state_b)
        self.h_state_b = Variable(self.h_state_b.data)

        print(pred_y_a)
        print(pred_y_b)

        command = Command()
#        command.accelerator = to_0_1(pred_y_a.data[0])
#        command.brake = to_0_1(pred_y_a.data[1])
        command.accelerator = put_in_range(0.0,1.0, pred_y_a.data.numpy().flatten()[0])
        command.brake = put_in_range(0.0,1.0, pred_y_a.data.numpy().flatten()[1])

        if command.accelerator > 0:
            if carstate.rpm > 8000:
                command.gear = carstate.gear + 1
        # else:
        #     command.brake = min(-acceleration, 1)
        if carstate.rpm < 2500:
            command.gear = carstate.gear - 1

        if not command.gear:
            command.gear = carstate.gear or 1


        command.steering = put_in_range(-1.0,1.0,pred_y_b.data.numpy().flatten()[0])
        command.focus = 0.0



        print("accelerator: " + str(pred_y_a.data.numpy().flatten()[0]) + "," + str(command.accelerator))
        print("brake: " +str(pred_y_a.data.numpy().flatten()[1])+","+str(command.brake))
        print("gear: " + str(command.gear))
        print("steering: " + str(pred_y_b.data.numpy().flatten()[0]))
        print("focus:" +str(command.focus))
        print("------------")


        if self.data_logger:
            self.data_logger.log(carstate, command)


        return command


