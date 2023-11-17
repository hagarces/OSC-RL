import torch
from copy import deepcopy
import os
import matplotlib.pyplot as plt
import numpy as np


def save_net(file_name, net, net_class, overwrite = False):
    checkpoint = {'model': net_class(*net.get_args()),
                  'state_dict': net.state_dict()}
    
    name_addition = ""
    if not overwrite:
        warning_cont = 1
        while True:
            if (file_name + name_addition + '.pth') in os.listdir():
                warning_cont += 1
                name_addition = "V" + str(warning_cont)           
            else:
                if warning_cont > 1:
                    file_name += name_addition
                    print("[WARNING]: that name is already taken!" + 
                          " it will be saved as V" + str(
                        warning_cont))
                break
    
    torch.save(checkpoint, file_name + '.pth')
    return name_addition

def load_net(filepath):
    checkpoint = torch.load(filepath + ".pth")
    model = checkpoint['model']
    model.load_state_dict(deepcopy(checkpoint['state_dict']))
    return model

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau)

def plot_stats(stats_history, epochs, epoch_interval):
    stats_name = ["Score", "Goal %", "Dist", "Collision", "Energy"]
    x = np.arange(1, int(epochs/epoch_interval)) * epoch_interval

    for i in range(len(stats_history)):
        y = np.array(stats_history[i])
        fig = plt.figure(figsize=(8, 5))
        plt.plot(x, y)
        plt.xlabel('episodes')
        plt.ylabel(stats_name[i])
        plt.show()