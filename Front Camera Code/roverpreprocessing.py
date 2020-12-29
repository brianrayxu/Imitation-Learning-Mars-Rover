import numpy as np
#from analyse import data_analyse_seven
from roverimitations import load_imitations
import random
import os
import matplotlib.pyplot as plt

def data_reduce(data_folder):
    # """
    # Reduce data to balance the distribution
    # """
    observations, actions = load_imitations(data_folder)
    # select = []
    # for index in range(len(actions)):
    #     if actions[index][0] == 0 and actions[index][1] > 0:     # PUT TYPE OF DATA THAT IS IN EXCESS
    #         select.append(index)
    # select = select[:int(len(select)/2)]
    # actions = np.delete(actions, select, 0)
    # observations = np.delete(observations, select, 0)
    # #print("len_action= ", actions.shape)
    # print("========= Done!, " + str(int(len(select))) + \
    #     " Acc have been deleted ========")
    # print("====================================================")


#REPEAT FOR KEEP
    select = []
    
    for index in range(len(actions)):
        if(actions[index][0] == 0 and actions[index][1] == 0 and actions[index][2] == 0):     # PUT TYPE OF DATA THAT IS IN EXCESS
            select.append(index)
    random.shuffle(select)
    select = select[:int(len(select)/2)]
    actions = np.delete(actions, select, 0)
    observations = np.delete(observations, select, 0)
    #print("len_action= ", actions.shape)
    print("========= Done!, " + str(int(len(select))) + \
        " Acc have been deleted ========")
    print("====================================================")
    return observations, actions

def plot_acc(data1, name):
    acc1 = np.load(data1, allow_pickle = True)

    plt.figure()
    plt.plot(range(len(acc1)), acc1, label=name)

    plt.title('Training Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    # plt.show()
    plt.savefig(str(name)+'.png')


plot_acc('roverloss.npy', 'roverlossv1')


if __name__ == "__main__":
    print("Be careful before running this file")
    #if  sys.argv[1] == "prep":
        #data_analyse_seven('data/teacher')
        #data_reduce('data/teacher')
        #data_analyse_seven('data/teacher')
    #else :
    # directory = "C:\\Users\\brian\\Downloads\\RoboND-Python-StarterKit\\RoboND-Rover-Project\\gitcopy\\EC500_project\\"
    # print('sample test')
    # observations, actions = load_imitations(directory)
    # #print("observations[0]: ", observations[0])
    # print("observations shape: ", observations.shape)
    # actions = np.delete(actions,0, 0)
    # observations = np.delete(observations, 0, 0)
    # #print("observations[0]: ", observations[0])
    # print("observations shape: ", observations.shape)
    


