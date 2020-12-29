import numpy as np
import matplotlib.pyplot as plt
from roverimitations import load_imitations

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

directory = "C:\\Users\\brian\\Downloads\\RoboND-Python-StarterKit\\RoboND-Rover-Project\\gitcopy\\EC500_project\\"
data_analyse_seven(directory)