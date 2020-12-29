import numpy as np

# added later
import os
import torch

directory = "C:\\D drive\\Fall 2020\\EC500\\project\\EC500_project\\code\\"
trained_network_file = os.path.join(directory, 'data\\train.t7')

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

# This is where you can build a decision tree for determining throttle, brake and steer
# commands based on the output of the perception_step() function
def decision_step(Rover):
    # update action based on current observation
    infer_action = torch.load(trained_network_file, map_location='cuda') # original 'cpu'
    infer_action.eval()
    device = torch.device('cuda')   # ofiginal 'cpu'
    infer_action = infer_action.to(device)
    # observation = Rover.img # original front camera Image
    observations = get_vision_img(Rover.img)
    action_scores = infer_action(torch.Tensor(
        np.ascontiguousarray(observation[None])).to(device))
    Rover.steer, Rover.throttle, Rover.brake = infer_action.scores_to_action(action_scores)
    # print("Rover.steer=",Rover.steer, "Rover.throttle=", Rover.throttle, "Rover.brake=", Rover.brake)
    return Rover
