from pytocl.driver import Driver
from pytocl.car import State, Command


##
import os
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


class torcs_mlp_NN_acc(nn.Module):

    def __init__(self, num_features, num_nodes_hidden, num_output):
        super(torcs_mlp_NN_acc, self).__init__()

        # 2 hidden layer
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

        # 2 hidden layer
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

        # 2 hidden layer
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

        self.step_count=0
        self.info_filename = "file" + str(np.random.randint(1000000000)) + ".swarmfile"
        self.info_file = open(self.info_filename, "w")

        self.drive_aggressive = 0
        self.no_turn = 0 # 0 = false, 1 = true
        self.go_forward = 0   #go forward because you learned it is a straight now

        trained_torcs_NN_filename_acc = "pytocl/torcs_NN_trained_acc_new_q"
        trained_torcs_NN_filename_brake = "pytocl/torcs_NN_trained_brake_new_q"
        trained_torcs_NN_filename_steer = "pytocl/torcs_NN_trained_steer_new_good_22_input"
        num_features = 22
        num_nodes_hidden = 100
        self.trained_torcs_NN_acc = torcs_mlp_NN_acc(1, 20, 1)
        self.trained_torcs_NN_acc.load_state_dict(torch.load(trained_torcs_NN_filename_acc))
        self.trained_torcs_NN_brake = torcs_mlp_NN_brake(3, 20, 1)
        self.trained_torcs_NN_brake.load_state_dict(torch.load(trained_torcs_NN_filename_brake))
        self.trained_torcs_NN_steer = torcs_mlp_NN_steer(22, 20, 1)
        self.trained_torcs_NN_steer.load_state_dict(torch.load(trained_torcs_NN_filename_steer))


    def drive(self, carstate: State) -> Command:
        """
        Produces driving command in response to newly received car state.

        This is a dummy driving routine, very dumb and not really considering a
        lot of inputs. But it will get the car (if not disturbed by other
        drivers) successfully driven along the race track.
        """
#
# Setting: we have multiple individuals along with other opponents.
# our goal: all individuals try to follow a "chain",
# by trying to connect to a teammate which is in front of you.
#
# information rules:
#  - every 100k steps, write pos and "experience info" of the individual to file
#  - every 100k+50 steps, read info_files of the swarm
#
# -> with the info, all individuals in the swarm knows the position
# of everyone on the track.
# Behaviour:
#  - the frontrunner because he is in front will do nothing.
#  - for individuals not in the front, they can check whether the one in
# front is a individual of the swarm or not.
# if it is, then "it is connected to your teammate", so just continue driving
# if the one in front is not your teammate, it means you atleast have to get
# past him to get to the frontrunner.  <- idea is attraction to frontrunner
#
# to do this, the individuals in the back will make use of the info the swarm
# produces (especially the frontrunner) <- idea is ants with pheromone, everyone
# "drops" every certain amount of time some information about their last action.
#
# Each individual writes to the info_file his:
# current position, dist_from_start, number of "turns" he made in last few steps
#
# using the current position, each indivual can calculate his relative position across
# all racers
# using dist_from_start, each individual can calculate whether the number of "turns"
# is relevant for his current part of the track
# using number of "turns", each individual can determine whether the coming part
# of the track is straight and in the case of yes, it can thus accelerate without
# being afraid that there could have been a corner.
# (normally an individual can only look 200 dist in front of him, but now
# he is able to look much more further ahead.)
#
#
#
#
#
#


        length_of_straight=0
        own_position = carstate.race_position
        own_dist_from_start = carstate.distance_from_start

        # every 100 steps, write info in own file
        if (self.step_count % 50) == 0:
            print("now 100k")
            # write own race position to file
            info_to_write = str(own_position) + "," + \
                            str(own_dist_from_start) + "," + \
                            str(self.no_turn) + "\n"
            self.info_file.write(info_to_write)
            self.info_file.flush()

            self.no_turn = 1

        # w
        check_points=[] ; no_turn_list = []
        if ((self.step_count-25) % 50) == 0:
            print("now 100k+50")
            # read info_file of teammate
            for qfile in os.listdir("."):
                if qfile.endswith(".swarmfile"):
                    # no need to read own info_file
                    if qfile != self.info_filename:
                        teammate_info_file = open(qfile, "r")
                        all_info = teammate_info_file.read().strip().split("\n")
                        print("all_info is: " + str(all_info))
                        for line in all_info:
                            print("line is: " + str(line))
                            infos_read = line.split(",")
                            teammate_pos = int(infos_read[0])
                            dist_f_s = float(infos_read[1])
                            n_t = int(infos_read[2])
                            check_points.append(dist_f_s)
                            no_turn_list.append(n_t)
            print("@@@@")
            # check pos
            if own_position > teammate_pos:
                # you are behind your teammate
                if own_position - 1 == teammate_pos:
                    # you are exactly 1 pos behind teammate, so do nothing
                    print("im connected")
                    self.go_forward = 0
                else:
                    # you are atleast 2 places behind teammate
                    for index in range(0,len(check_points)-1):
                        print("own, cp index, cp index+1: " + str(own_dist_from_start) + ","+str(check_points[index]) + ","+str(check_points[index+1]) + ",")
                        if own_dist_from_start > check_points[index] and \
                            own_dist_from_start < check_points[index+1]:
                            print("cp[index:] : "+str(check_points[index:]))
                            # calc how long straight is beginning from check_point[index+1]
                            # to the last checkpoint.
                            length_of_straight = 0
                            for k in range(index+1, len(no_turn_list)):
                                print("no_turn_list , index:" + str(no_turn_list) + ", " + str(index))
                                if no_turn_list[k] == 0:
                                    # is dist of straight
                                    print("k is: " + str(k))
                                    length_of_straight = check_points[k-1] - own_dist_from_start
                                    break
                                elif k == len(no_turn_list)-1:
                                    print("k is: " + str(k) + "," + str(check_points[k] - own_dist_from_start))
                                    length_of_straight = check_points[k] - own_dist_from_start
#                                else:
#                                    length_straight=length_straight+1

                            print("length_of_straight is: " + str(length_of_straight))

                            self.go_forward = int(length_of_straight/2)

                            break
            else:
                # you are in front, keep on going
                print("im in front")
                print("own distfromstart: " + str(own_dist_from_start))
                if own_position == teammate_pos - 1:
                    # teammate is behind you, drive more aggresive now
                    self.drive_aggressive = 1
                else:
                    self.drive_aggressive = 0

        self.step_count = self.step_count+1

        #print(" " + str(self.qqq))
        #print("------------")
        #print("race position: " + str(carstate.race_position))
        #print("dist from start: " + str(carstate.distance_from_start))
        #print("opponents: " + str(carstate.opponents))
        #print("speed_x: " + str(carstate.speed_x))
        #print("distance_from_center: " + str(carstate.distance_from_center))
        #print("angle: " + str(carstate.angle))
        #print("distances_from_edge: ")
        #print(carstate.distances_from_edge)
        #print("------------")
        #print(carstate.distances_from_egde_valid)


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



        test_x = np.zeros(22)
        test_x[0] = carstate.speed_x
        test_x[1] = carstate.distance_from_center
        test_x[2] = carstate.angle
        test_x[3:] = np.array(carstate.distances_from_edge)
        test_x = Variable(torch.Tensor(test_x))

        new_x = np.zeros(3)
        new_x[0] = self.prev_gear
        new_x[1] = carstate.speed_x
        new_x[2] = carstate.distances_from_edge[9]
        new_x = Variable(torch.Tensor(new_x))

        pred_y_acc = self.trained_torcs_NN_acc(Variable(torch.Tensor([self.prev_gear])))
        pred_y_brake = self.trained_torcs_NN_brake(new_x)
        pred_y_steer = self.trained_torcs_NN_steer(test_x)

        #print(pred_y_acc)
        #print(pred_y_brake)
        #print(pred_y_steer)

        command = Command()
        command.accelerator = put_in_range(0.0,1.0,pred_y_acc.data[0])
        command.brake = put_in_range(0.0,1.0, pred_y_brake.data[0])
        

        if command.accelerator > 0:
            if carstate.rpm > 8000:
                command.gear = carstate.gear + 1
        # else:
        #     command.brake = min(-acceleration, 1)
        if carstate.rpm < 2500:
            command.gear = carstate.gear - 1

        if not command.gear:
            command.gear = carstate.gear or 1

        command.steering = put_in_range(-1.0,1.0,pred_y_steer.data[0])
        command.focus = 0.0

        #print("accelerator: " + str(pred_y_acc.data[0]) + "," + str(command.accelerator))
        #print("brake: " +str(pred_y_brake.data[0]))
        #print("gear: " + str(command.gear))
        #print("steering: " + str(pred_y_steer.data[0]))
        #print("focus:" +str(command.focus))
        #print("------------")

        if abs(command.steering) > 0.1:
            self.no_turn = 0

        if self.data_logger:
            self.data_logger.log(carstate, command)

        # remember gear
        self.prev_gear = command.gear


        # if you know there are atleast 2 strai
        # if you know that there is a straight di
        if self.go_forward>0:
            print("overriding NN, go_forw: " + str(self.go_forward))
            command.accelerator = 1.0
            self.go_forward=self.go_forward-1

        if self.drive_aggressive == 1:
            #print("aggression now")
            if command.accelerator < 1.0:#0.5:
                print("aggression")
                command.accelerator=1.0#0.5
            command.brake = 0.7*command.brake

        return command


