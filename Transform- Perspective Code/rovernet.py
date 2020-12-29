import torch
import torch.nn as nn
import numpy as np

class ClassificationNetwork(torch.nn.Module):
    def __init__(self):
        """
        1.1 d)
        Implementation of the network layers. The image size of the input
        observations is 96x96 pixels.
        """
        super().__init__()

        if torch.cuda.is_available():
            dev = "cuda" #"cuda:0"
            print("working on gpu")
        else:
            dev = "cpu"
            print("working on cpu")

        device = torch.device(dev)

        #=================================================================
        # EasyNet : 2 Conv2d and 2 Linear
        # 7 classes
        self.conv1 = nn.Conv2d(3, 16, 5, stride=4)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(0.5)
        self.fc1 = nn.Linear(23712, 128)
        self.fc2 = nn.Linear(128, 7)    # modified to 7 classes

    def normalization(self, x):
        """
        This funtion is used to normalize data based on the Max-Min method,
        the return value must between [-1, 1]
        x:        list of size len(x)
        return:   list of size len(x)
        """
        if len(x) > 1:
            x = ((x - min(x)) / (max(x) - min(x)))
        return x

    def forward(self, observation):
        """
        1.1 e)
        The forward pass of the network. Returns the prediction for the given
        input observation.
        observation:   torch.Tensor of size (batch_size, 96, 96, 3)
        return         torch.Tensor of size (batch_size, number_of_classes)
        """
        #=================================================================
        #====================== Data Preprocessing =======================
        #=================================================================
        # Convert RGB to Grayscale
        # rgb2gray = 0.2989*observation[:, :, :, 0] + \
        #            0.5870*observation[:, :, :, 1] + \
        #            0.1140*observation[:, :, :, 2]

        x = torch.reshape(observation, (-1, 160, 320, 3))
        #x = x.permute(0, 3, 1, 2)
        #=================================================================
        # *** This part is used to add the sensor input ***
        #        uncommand with line 185 together
        # extract_input = self.extract_input(observation)
        # extract_input = self.fc3(extract_input)
        #=================================================================
        # EasyNet for 7 classes
        x = observation.permute(0, 3, 1, 2)
        x = self.act(self.conv1(x))
        x = self.drop(x)
        x = self.act(self.conv2(x))
        x = self.drop(x)

        #check input to linear layer
        # print("shape: ", x.size(0))

        x = x.reshape(x.size(0), -1)
        # x = torch.cat([x, extract_input], dim=1)  # uncommand line 175,176 before
        x = self.fc1(x)
        x = self.fc2(x)

        return x

    def actions_to_classes(self, actions):
        """
        For a given set of actions map every action to its corresponding
        action-class representation. Assume there are number_of_classes
        different classes, then every action is represented by a
        number_of_classes-dim vector which has exactly one non-zero entry
        (one-hot encoding). That index corresponds to the class number.
        actions:        python list of N torch.Tensors of size 3
        return          python list of N torch.Tensors of size number_of_classes
        """
        movement = []

        """
        # 7 Classes Classification [steering, throttle, brake]
        for action in actions:
            if action[0] == 0 and action[1] > 0:                         # throttle
                movement.append(torch.Tensor([1, 0, 0, 0, 0, 0, 0]))
            elif action[0] == 0 and action[2] > 0:                        # brake
                movement.append(torch.Tensor([0, 1, 0, 0, 0, 0, 0]))
            elif action[0] > 0 and action[1] == 0:                       # left
                movement.append(torch.Tensor([0, 0, 1, 0, 0, 0, 0]))
            elif action[0] < 0 and action[1] == 0:                        # right
                movement.append(torch.Tensor([0, 0, 0, 1, 0, 0, 0]))
            elif action[0] > 0 and action[1] > 0:                       # throttle left
                movement.append(torch.Tensor([0, 0, 0, 0, 1, 0, 0]))
            elif action[0] < 0 and action[1] > 0:                       # throttle right
                movement.append(torch.Tensor([0, 0, 0, 0, 0, 1, 0]))
            elif action[0] == 0 and action[1] == 0 and action[2] == 0:     # keep - modified to 7 classes
                movement.append(torch.Tensor([0, 0, 0, 0, 0, 0, 1]))
        """
        #=================================================================
        # 4 Classes Classification [steering, throttle, brake]
        for action in actions:
            if action[0] == 0 and action[1] > 0:                         # throttle
                movement.append(torch.Tensor([1, 0, 0, 0]))
            elif action[0] > 0 and action[1] > 0:                       # throttle left
                movement.append(torch.Tensor([0, 1, 0, 0]))
            elif action[0] < 0 and action[1] > 0:                       # throttle right
                movement.append(torch.Tensor([0, 0, 1, 0]))
            elif action[0] == 0 and action[1] == 0 and action[2] == 0:     # keep - modified to 7 classes
                movement.append(torch.Tensor([0, 0, 0, 1]))
        #=================================================================

        return movement

    def scores_to_action(self, scores):
        """
        Maps the scores predicted by the network to an action-class and returns
        the corresponding action [steer, gas, brake].
        scores:         python list of torch.Tensors of size number_of_classes
        return          (float, float, float)
        """
        max_index = torch.argmax(scores, 1)
        #=================================================================
        # 7 Classes Classification [steering, throttle, brake] - for vision_img
        """
        if max_index == 0:
            action = [0.0, 0.6, 0.0] #throttle
        elif max_index == 1:
            action = [0.0, 0.0, 1.0] #brake
        elif max_index == 2:
            action = [5.0, 0.0, 0.0] #left
        elif max_index == 3:
            action = [-5.0, 0.0, 0.0] #right
        elif max_index == 4:
            action = [10.0, 0.5, 0.0] # throttle left
        elif max_index == 5:
            action = [-10.0, 0.5, 0.0] # throttle right
        elif max_index == 6:
            action = [0.0, 0.0, 0.0]     # keep - modified to 7 classes
        """
        #=================================================================
        # 4 Classes Classification [steering, throttle, brake] - for vision_img
        if max_index == 0:
            action = [0.0, 0.6, 0.0] # throttle
        elif max_index == 1:
            action = [10.0, 0.5, 0.0] # throttle left
        elif max_index == 2:
            action = [-10.0, 0.5, 0.0] # throttle right
        elif max_index == 3:
            action = [0.0, 0.0, 0.0]     # keep - modified to 7 classes
        #=================================================================

        return float(action[0]), float(action[1]), float(action[2])
