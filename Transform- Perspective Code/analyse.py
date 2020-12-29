# Copyright (c) 2020 Yanyu Zhang zhangya@bu.edu All rights reserved.
import numpy as np
import matplotlib.pyplot as plt
from roverimitations import load_imitations


def plot_acc(data1, name):
    acc1 = np.load(data1, allow_pickle = True)  # why "allow_pickle = True"
                                                # (online: “Pickling” is the process whereby
                                                # a Python object hierarchy is converted into a byte stream)
    plt.figure()
    plt.plot(range(len(acc1)), acc1, label=name)

    plt.title('Training Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    # plt.show()
    plt.savefig('images/'+str(name)+'.png')

# plot_acc('result/easy_gray.npy', 'MyNet')     # do we only need to put easy_gray.npy file in the "result" folder?

def data_analyse_seven(data_folder):
    observations, actions = load_imitations(data_folder)
    left, left_throttle, right, right_throttle, acc, brakes, keep = 0,0,0,0,0,0,0
    for action in actions: # [steer, throttle, brake]
        if action[0] > 0 and action[1] == 0:       # left
            left += 1
        elif action[0] > 0 and action[1] > 0:      # left throttle
            left_throttle += 1
        elif action[0] < 0 and action[1] == 0:     # right
            right += 1
        elif action[0] > 0 and action[1] > 0:      # right throttle
            right_throttle += 1
        elif action[0] == 0 and action[1] > 0:     # throttle - accelerate
            acc += 1
        elif action[0] == 0 and action[2] > 0:     # brake
            brakes += 1
        elif action[0] == 0 and action[1] == 0 and action[2] == 0:     # keep   #  why do we need "keep" folder?
            keep += 1
    summ = left+left_throttle+right+right_throttle+acc+brakes+keep
    print("====================================================")
    print("------------------------ Data_analyse_seven ------------------------")
    print("----------- Data pairs in total =", str(len(actions)), "------------")
    print("----------- Data pairs be used =", str(summ), "-------------")
    print("====================================================")
    print("Left = ", left)
    print("Left_throttle = ", left_throttle)
    print("Right = ", right)
    print("Right_throttle = ", right_throttle)
    print("Accelerate = ", acc)
    print("Break = ", brakes)
    print("Keep = ", keep)
    print("====================================================")

def data_analyse_ten(data_folder):
    observations, actions = load_imitations(data_folder)
    left, left_throttle, left_brake, right, right_throttle, right_brake, throttle_forward, throttle_backward, brakes, keep = 0,0,0,0,0,0,0,0,0,0
    for action in actions: # [steer, throttle, brake]
        if action[0] > 0 and action[1] == 0:       # left
            left += 1
        elif action[0] > 0 and action[1] > 0:      # left throttle
            left_throttle += 1
        elif action[0] > 0 and action[2] > 0:      # left brake
            left_brake += 1
        elif action[0] < 0 and action[1] == 0:     # right
            right += 1
        elif action[0] < 0 and action[1] > 0:      # right throttle
            right_throttle += 1
        elif action[0] > 0 and action[2] > 0:      # right brake
            right_brake += 1
        elif action[0] == 0 and action[1] > 0:     # throttle_forward
            throttle_forward += 1
        elif action[0] == 0 and action[1] < 0:     # throttle_backward
            throttle_backward += 1
        elif action[0] == 0 and action[2] > 0:     # brake
            brakes += 1
        elif action[0] == 0 and action[1] == 0 and action[2] == 0:     # keep   #  why do we need "keep" folder?
            keep += 1
        else:
            print("[steer, throttle, brake]: [", action[0],", ", action[1],", ", action[2], "]")
    summ = left+left_throttle+left_brake+right+right_throttle+right_brake+throttle_forward+throttle_backward+brakes+keep
    print("====================================================")
    print("------------------------ Data_analyse_ten -------------------------")
    print("----------- Data pairs in total =", str(len(actions)), "------------")
    print("----------- Data pairs be used =", str(summ), "-------------")
    print("====================================================")
    print("Left = ", left)
    print("Left_throttle = ", left_throttle)
    print("Left_brake = ", left_brake)
    print("Right = ", right)
    print("Right_throttle = ", right_throttle)
    print("Right_brake = ", right_brake)
    print("Accelerate Forward= ", throttle_forward)
    print("Accelerate Backward= ", throttle_backward)
    print("Break = ", brakes)
    print("Keep = ", keep)
    print("====================================================")

data_directory = "C:\\D drive\\Fall 2020\\EC500\\project\\EC500_project\\code\\data\\"
# data_analyse_seven(data_directory)
data_analyse_ten(data_directory)
