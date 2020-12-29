import os
import numpy as np

# added
import pandas as pd
from PIL import Image


def load_imitations(data_folder):
    """
    Given the folder containing the expert imitations, the data gets loaded and
    stored it in two lists: observations and actions.
    Parameter:
        data_folder: python string, the path to the folder containing the
                    actions csv files & observations jpg files
    return:
        observations:   python list of N numpy.ndarrays of size (160, 320, 3)
        actions:        python list of N numpy.ndarrays of size 3
        (N = number of (observation, action) - pairs)
    """
    # get actions
    csv_file = pd.read_csv(data_folder+'robot_log.csv', sep=';',header=None)
    csv_arr = csv_file.values
    actions = np.asarray(csv_arr[1:, 1:4], dtype=np.float16)
    #print("actions[0]: ", actions[0], "; action[1]:", actions[1])
    #print(actions.dtype)
    # get observations
    obs_files = os.listdir(data_folder + '/IMG/')
    observations = [0]*int(len(obs_files))  # create list
    index = 0
    for filename in obs_files:  # loop through all files
        open_file_name = os.path.join(os.path.join(data_folder + '/IMG/'), filename)
        observations[index] = np.asarray(Image.open(open_file_name))
        index += 1
    observations = np.asarray(observations, dtype=np.float16)
    #print("observations[0]: ", observations[0])
   # print("observations[0] shape: ", observations[0].shape)

    return observations, actions

# following code is for testing purpose only, need to be commented out later
#data_folder = "C:\\Users\\brian\\Downloads\\RoboND-Python-StarterKit\\RoboND-Rover-Project\\gitcopy\\EC500_project\\"
#data_folder = '/Users/hairuosun/Library/Mobile Documents/com~apple~CloudDocs/BU/Fall 2020/Courses/EC 500 A2/Project/Github Simulation/EC500_project/data/'
#load_imitations(data_folder)