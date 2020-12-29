import os
import numpy as np

# added
import pandas as pd
from PIL import Image
import cv2

# Define a function to perform a perspective transform
def perspect_transform(img, src, dst):

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))# keep same size as input image

    mask = cv2.warpPerspective(np.ones_like(img[:,:,0]), M, (img.shape[1], img.shape[0]))
    return warped, mask

# Identify pixels above the threshold
# Threshold of RGB > 160 does a nice job of identifying ground pixels only
def color_thresh(img, rgb_thresh=(160, 160, 160)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (img[:,:,0] > rgb_thresh[0]) \
                & (img[:,:,1] > rgb_thresh[1]) \
                & (img[:,:,2] > rgb_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = 1
    # Return the binary image
    return color_select

def find_rocks(img, levels=(110, 110, 50)):
    rockpix = ((img[:,:,0] > levels[0]) \
                & (img[:,:,1] > levels[1]) \
                & (img[:,:,2] < levels[2]))

    color_select = np.zeros_like(img[:,:,0])
    color_select[rockpix] = 1

    return color_select

def get_vision_img(image):
    vision_image = np.zeros((160, 320, 3), dtype=np.float)
    # Perform perception steps to update Rover()
    # TODO:
    # NOTE: camera image is coming to you in Rover.img
    # 1) Define source and destination points for perspective transform
    # 2) Apply perspective transform
    # Define calibration box in source (actual) and destination (desired) coordinates
    # These source and destination points are defined to warp the image
    # to a grid where each 10x10 pixel square represents 1 square meter
    # The destination box will be 2*dst_size on each side
    dst_size = 5
    # Set a bottom offset to account for the fact that the bottom of the image
    # is not the position of the rover but a bit in front of it
    # this is just a rough guess, feel free to change it!
    bottom_offset = 6
    source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
    destination = np.float32([[image.shape[1]/2 - dst_size, image.shape[0] - bottom_offset],
                      [image.shape[1]/2 + dst_size, image.shape[0] - bottom_offset],
                      [image.shape[1]/2 + dst_size, image.shape[0] - 2*dst_size - bottom_offset],
                      [image.shape[1]/2 - dst_size, image.shape[0] - 2*dst_size - bottom_offset],
                      ])
    warped, mask = perspect_transform(image, source, destination)
    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    threshed = color_thresh(warped)
    obs_map = np.absolute(np.float32(threshed) - 1) * mask
    # 4) Update vision_image (this will be displayed on left side of screen)
        # Example: vision_image[:,:,0] = obstacle color-thresholded binary image
        #          vision_image[:,:,1] = rock_sample color-thresholded binary image
        #          vision_image[:,:,2] = navigable terrain color-thresholded binary image
    vision_image[:,:,2] = threshed * 255
    vision_image[:,:,0] = obs_map * 255

    # 5) See if we can find some rocks
    rock_map = find_rocks(warped, levels = (110, 110, 50))
    if rock_map.any():
        vision_image[:, :, 1] = rock_map * 255
    else:
        vision_image[:, :, 1] = 0

    return vision_image

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
    actions = np.asarray(csv_arr[:, 1:4], dtype=np.float16)
    #print("actions[0]: ", actions[0], "; action[1]:", actions[1])

    #print(actions.dtype)

    # get observations
    obs_files = os.listdir(data_folder + '\\IMG\\')
    observations = [0]*int(len(obs_files))  # create list
    index = 0
    for filename in obs_files:  # loop through all files
        open_file_name = os.path.join(os.path.join(data_folder + '/IMG/'), filename)
        # originally: observations[index] = np.asarray(Image.open(open_file_name))
        # now:
        front_cam_img = np.asarray(Image.open(open_file_name))
        observations[index] = get_vision_img(front_cam_img)
        index += 1
    observations = np.asarray(observations)
    #print("observations[0]: ", observations[0])
    # print("observations[0] shape: ", observations[0].shape)

    return observations, actions

# following code is for testing purpose only, need to be commented out later
# data_folder = '/Users/hairuosun/Library/Mobile Documents/com~apple~CloudDocs/BU/Fall 2020/Courses/EC 500 A2/Project/Github Simulation/EC500_project/data/'
# load_imitations(data_folder)
