# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 18:45:02 2019

@author: or_ra
"""

import numpy as np
import random
from itertools import combinations
import torch
import copy 
from skimage.transform import rescale

def divide_learners(AllActions, AllAngles, AllImage, time):
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
        
        learning_dataset["camera t-1"] =  torch.FloatTensor(np.abs(AllImage[time-1]))
        learning_dataset["camera t"] = torch.FloatTensor(np.abs(AllImage[time]))
        learning_dataset["camera angle t-1"] = torch.tensor((AllAngles[time-1])[1])
        learning_dataset["camera angle t"] = torch.tensor((AllAngles[time])[1])
        learning_dataset["arm angle t-1"] = torch.tensor((AllAngles[time-1])[0])
        learning_dataset["arm angle t"] = torch.tensor((AllAngles[time])[0])
        learning_dataset["camera action t"] = torch.tensor((AllActions[time])[1])
        learning_dataset["arm action t"] = torch.tensor((AllActions[time])[0])
    
    comb_step = []
    for a, b in combinations(learning_dataset.keys(), 2):
        comb_step.append([a, b])     
    
    combinations_list = []
    for name, val_y in learning_dataset.items():
        for comb in comb_step:
            if name not in comb:
                combinations_list.append([comb, name])
                
                
                
    
    return learning_dataset, combinations_list 

def learners_nets():
    '''
    fc_netSSS, fc_netSSC, conv_netCSS, conv_netCCS, conv_netSSC, conv_netCSC
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
                net_dict[name[0][0],name[0][1], name[1], 'CCS'] = 1#copy.deepcopy(conv_netCCS)
            elif "camera t" in name:
                net_dict[name[0][0],name[0][1], name[1], 'CSC'] = 2#copy.deepcopy(conv_netCSC)
            else:
                net_dict[name[0][0],name[0][1], name[1], 'CSS'] = 3#copy.deepcopy(conv_netCSS)
        if "camera t" in name[0]:
            if "camera t-1" in name:
                net_dict[name[0][0],name[0][1], name[1], 'CSC'] = 2#copy.deepcopy(conv_netCSC)
            else:
                net_dict[name[0][0],name[0][1], name[1], 'CSS'] = 3#copy.deepcopy(conv_netCSS)
        else:
            if "camera t" in name:
                net_dict[name[0][0],name[0][1], name[1], 'SSC'] = 4#copy.deepcopy(fc_netSSC)
            if "camera t-1" in name:
                net_dict[name[0][0],name[0][1], name[1], 'SSC'] = 4#copy.deepcopy(fc_netSSC)
            else:
                net_dict[name[0][0],name[0][1], name[1], 'SSS'] = 5#copy.deepcopy(fc_netSSS)
        

    return net_dict
                
def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return torch.from_numpy(np.eye(num_classes, dtype='uint8')[y])
    
    
#Dividing the image into singular pixals to learn for the first layer

def create_pixel_vecCC(imaget, imagetm1, sensor_output, rescale_image = False, binary_action = False):
    '''
    This will take the two images from the camera at time t and t-1
    and will output a tensor of size [HxWxC, 2].
    To be the input of the fully conected network.
    Rescale is defult to no, if we choose to rescale it will rescale by factor 
    6
    '''
    imaget, imagetm1 = imaget.data.numpy(), imagetm1.data.numpy()
    
    if rescale_image:
        imaget = rescale(imaget, 1.0/6.0, anti_aliasing=True, multichannel=True)
        imagetm1 = rescale(imagetm1, 1.0/6.0, anti_aliasing=True, multichannel=True)
    
    
    imaget, imagetm1 = torch.from_numpy(imaget), torch.from_numpy(imagetm1)    
    pixel_vecCC = torch.empty([imaget.size()[0] * imaget.size()[1] * imaget.size()[2], 2])
    target = torch.empty([imaget.size()[0] * imaget.size()[1] * imaget.size()[2], 1])
    
    imaget = torch.reshape(imaget, [imaget.size()[0] * imaget.size()[1] * imaget.size()[2]])
    imagetm1 = torch.reshape(imagetm1, [imagetm1.size()[0] * imagetm1.size()[1] * imagetm1.size()[2]])
    
    if binary_action:
        if sensor_output != 0:
            sensor_output = torch.tensor([1])
        else:
            sensor_output = torch.tensor([0])
            
    pixel_vecCC[:, 0] = imaget
    pixel_vecCC[:, 1] = imagetm1
    target[:,:] = sensor_output
    
    return pixel_vecCC, target

def create_pixel_vecCS(image_in, sensor_input, image_out, rescale_image = False, binary_action = False):
    '''
    This will take the images from the camera and a sensor input 
    and will output a tensor of size [HxWxC, 2].
    To be the input of the fully conected network.
    Rescale is defult to no, if we choose to rescale it will rescale by factor 
    6
    '''
    image_in, image_out = image_in.data.numpy(), image_out.data.numpy()
    if rescale_image:
        image_in = rescale(image_in, 1.0/6.0, anti_aliasing=True, multichannel=True)
        image_out = rescale(image_out, 1.0/6.0, anti_aliasing=True, multichannel=True)
    
    image_in = torch.from_numpy(image_in)
    image_out = torch.from_numpy(image_out)
    pixel_vecCS = torch.empty([image_in.size()[0] * image_in.size()[1] * image_in.size()[2], 2])
    
    image_in = torch.reshape(image_in, [image_in.size()[0] * image_in.size()[1] * image_in.size()[2]])
    image_out = torch.reshape(image_out, [image_out.size()[0] * image_out.size()[1] * image_out.size()[2]])

    if binary_action:
        if sensor_input != 0:
            sensor_input = torch.tensor([1])
        else:
            sensor_input = torch.tensor([0])
            
    pixel_vecCS[:, 0] = image_in
    pixel_vecCS[:, 1] = sensor_input
    
    target = image_out
    
    return pixel_vecCS, target 


def ChooseActionNoBrain(observation, percent_zero = 0.2):
    times, times2, times3, times4 = 0, 0, 0, 0
    f, y = 0.0, 0.0
    if observation[0][0]>0.9:
        times=50
        f = -10.0       
    
    elif times>0:
        f=-1.0        
        times += -1
    
    elif observation[0][0]<-0.9:
        f=10.0        
        times2 = 50
    
    elif times2>0:
        f=10.0        
        times2 += -1
                
    else:
        f = random.randrange(-2000,2000)/100
    if observation[0][1]>0.5:
        times3=19
        y = -20.0
    elif times3>0:
        y=-15.0
        times3 += -10

    
    elif observation[0][1]<-0.5:
        y=10.0
        times4 = 50
    elif times4>0:
        y=10.0
        times4 += -1
    else:
        y = random.randrange(-2000,2000)/100
    if random.randrange(0,100)/100 < percent_zero:
        f = 0.0
    if random.randrange(0,100)/100 < percent_zero:
        y = 0.0
    action = [f, y]
    return action    
    
    
    
    
    
    
    