# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 18:45:02 2019

@author: or_ra
"""

import numpy as np
from itertools import combinations
import copy
import torch

def divide_learners(action_set, AllAngles, AllImage, time):
    '''
    The function takes in the action set and the sensors inputs and returns a 
    cach of all the combinatorical possibilities for learning. 
    
    Argumants:
        action_set - a set of all actions from the enviorment. From t=0 to t. 
        sensor_set - a set of all sensors outputs, all the angles from the arm 
        camera plus the camera output. From time t=0 to t.
        time = The "time" step we are in, i.e. how many epochs the program ran.
    Returns:
        Learners - a cach 
        
    '''
    learning_dataset = {}
    if time == 0:
        learning_dataset = {}
    else:  
        
        learning_dataset["camera t-1"] =  torch.tensor(np.abs(AllImage[time-1]))
        learning_dataset["camera t"] = torch.tensor(np.abs(AllImage[time]))
        learning_dataset["camera angle t-1"] = torch.tensor((AllAngles[time-1])[1])
        learning_dataset["camera angle t"] = torch.tensor((AllAngles[time])[1])
        learning_dataset["arm angle t-1"] = torch.tensor((AllAngles[time-1])[0])
        learning_dataset["arm angle t"] = torch.tensor((AllAngles[time])[0])
        learning_dataset["camera action t"] = torch.tensor((action_set[time])[1])
        learning_dataset["arm action t"] = torch.tensor((action_set[time])[0])
    
    comb_step = []
    for a, b in combinations(learning_dataset.keys(), 2):
        comb_step.append([a, b])     
    
    combinations_list = []
    for name, val_y in learning_dataset.items():
        for comb in comb_step:
            if name not in comb:
                combinations_list.append([comb, name])
                
                
                
    
    return learning_dataset, combinations_list 

def learners_nets(fc_netSSS, fc_netSSC, conv_netCSS, conv_netCCS, conv_netSSC, conv_netCSC):
    '''
    A function that takes in the base models of the learners and creates a dict 
    holding relevent nets for relevent learners. Fully connected networks for 
    the action/sensors correlations and convelution networks for the correleations
    with one or more image inputs. 
    
    '''
    
    input_list = ["camera t-1", "camera t", "camera angle t-1", "camera angle t", 
                  "arm angle t-1", "arm angle t", "camera action t", "arm action t"]
    
    comb_list = []
    for a, b in combinations(input_list, 2):
        comb_list.append([a, b])
    
    nets_name = []
    for name in input_list:
        for comb in comb_list:
            if name not in comb:
                nets_name.append([comb, name])
                
    net_dict = {}
    for name in nets_name:
        if "camera t-1" in name[0]:
            if "camera t" in name[0]:
                net_dict[name[0][0],name[0][1], name[1]] = 1# copy.deepcopy(conv_netCCS)
            elif "camera t" in name:
                net_dict[name[0][0],name[0][1], name[1]] = 2 #copy.deepcopy(conv_netCSC)
            else:
                net_dict[name[0][0],name[0][1], name[1]] = 3 #copy.deepcopy(conv_netCSS)
        if "camera t" in name[0]:
            if "camera t-1" in name:
                net_dict[name[0][0],name[0][1], name[1]] = 5 #copy.deepcopy(conv_netCSC)
            else:
                net_dict[name[0][0],name[0][1], name[1]] = 6 #copy.deepcopy(conv_netCSS)
        else:
            if "camera t" in name:
                net_dict[name[0][0],name[0][1], name[1]] = 7 #copy.deepcopy(fc_netSSC)
            if "camera t-1" in name:
                net_dict[name[0][0],name[0][1], name[1]] = 8#copy.deepcopy(fc_netSSC)
            else:
                net_dict[name[0][0],name[0][1], name[1]] = 9 #copy.deepcopy(fc_netSSS)
        

    return net_dict
                
    
    
    
    
    
    
    
    
    
    
    