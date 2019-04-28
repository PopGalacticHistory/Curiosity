# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 18:31:02 2019

@author: or_ra
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pruning.layers import MaskedLinear, MaskedConv2d 

class LearnerFCNets(nn.Module):
    def __init__(self, D_in, D_fc1, D_fc2, D_out):
        super(LearnerFCNets, self).__init__()
        '''
        The learners of the engine actions and the angles senssors. 
        2 input action/senssor, real number between 1 and -1. 
        1 output action/senssor
        
        '''
        self.fc1 = MaskedLinear(D_in, D_fc1)
        self.bn1 = nn.BatchNorm1d(D_fc1)
        self.fc2 = MaskedLinear(D_fc1, D_fc2)
        self.bn2 = nn.BatchNorm1d(D_fc2)
        self.fc3 = MaskedLinear(D_fc2, D_out)
        
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        return torch.tanh(self.fc3(x))
    
    def set_masks(self, masks):
        # Should be a less manual way to set masks
        # Leave it for the future
        self.fc1.set_mask(masks[0])
        self.fc2.set_mask(masks[1])
        self.fc3.set_mask(masks[2])

class LearnerFCNets2(nn.Module):
    def __init__(self):
        super(LearnerFCNets2, self).__init__()
        '''
        The learners of the engine actions and the angles senssors. 
        2 input action/senssor, real number between 1 and -1. 
        1 output action/senssor
        
        '''
        self.fc1 = nn.Linear(2, 10)
        self.bn1 = nn.BatchNorm1d(10)
        self.fc2 = nn.Linear(10, 10)
        self.bn2 = nn.BatchNorm1d(10)
        self.fc3 = nn.Linear(10, 1)
        
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        return 20*torch.tanh(self.fc3(x))
    

class LearnerConvNetCSS(nn.Module):
    
    def __init__(self):
        super(LearnerConvNetCSS, self).__init__()
        '''
        The learner that has one or more camera inputs:
        CSS - Camera Sensor/action to Sensor/action
        The learners recieves: 
        1 input action/senssor, real number between 1 and -1
        1 input from camera, 320x240 rgb array 
        1 output action/senssor
       '''
        self.conv1 = nn.Conv2d(4, 10, kernel_size = 1, stride = 1)
        self.batchNorm1 = nn.BatchNorm2d(10, momentum=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(10, 30, kernel_size = 3, stride = 2)
        self.batchNorm1 = nn.BatchNorm2d(30, momentum=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(10, 60, kernel_size = 3, stride = 2)
        self.batchNorm1 = nn.BatchNorm2d(60, momentum=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(7*9*90, 2835)
        self.fc2 = nn.Linear(2835, 700)
        self.fc3 = nn.Linear(700, 250)
        self.fc4 = nn.Linear(250, 50)
        self.fc5 = nn.Linear(50, 1)
       
    def forward(self, x, y):
        if np.shape(x) == torch.Size([240, 320, 3]):
            image = x
            sensor = y
        else:
            image = y
            sensor = x
        
        sensor = sensor * torch.ones(240,320)
        sensor = sensor.byte()
        sensor = sensor.unsqueeze(2)
        combined = torch.cat((image, sensor), dim = 2)
        combined = combined.unsqueeze(0)
        combined = self.pool(F.relu(self.conv1(combined)))
        combined = self.pool(F.relu(self.conv2(combined)))
        combined = self.pool(F.relu(self.conv3(combined)))
        combined = combined.view(7*9*90)
        
        combined = F.relu(self.fc1(combined))
        combined = F.relu(self.fc2(combined))
        combined = F.relu(self.fc3(combined))
        combined = F.relu(self.fc4(combined))
        combined = F.relu(self.fc5(combined))
        
        return torch.tanh(combined)

class LearnerConvNetCCS(nn.Module):
    def __init__(self):
        super(LearnerConvNetCCS, self).__init__()
        '''
        The learner that has one or more camera inputs:
            CCS - Camera Camera to Sensor/action
            The learners recieves: 
                2 input from camera, 600x800 rgb array 
                1 output action/senssor
        '''
    
        self.conv1 = nn.Conv2d(3, 10, kernel_size = 1, stride = 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(10, 10, kernel_size = 1, stride = 1)
        self.conv3 = nn.Conv2d(10, 10, kernel_size = 1, stride = 1)
        self.fc1 = nn.Linear(10*50*75, 26250)
        self.fc2 = nn.Linear(26250, 13125)
        self.fc3 = nn.Linear(13125, 3281)
        self.fc4 = nn.Linear(3281, 410)
        self.fc5 = nn.Linear(410, 1)
       
    def forward(self, x, y):
        image1 = x
        image2 = y 
        image1 = self.pool(F.relu(self.conv1(image1)))
        image2 = self.pool(F.relu(self.conv1(image2)))
        image1 = self.pool(F.relu(self.conv2(image1)))
        image2 = self.pool(F.relu(self.conv2(image2)))
        image1 = self.pool(F.relu(self.conv3(image1)))
        image2 = self.pool(F.relu(self.conv3(image2)))
        image1 = image1.view(10*50*75)
        image2 = image2.view(10*50*75)
        combined_input = torch.cat((image1, image2), -1)
        combined_input = F.relu(self.fc1(combined_input))
        combined_input = F.relu(self.fc2(combined_input))
        combined_input = F.relu(self.fc3(combined_input))
        combined_input = F.relu(self.fc4(combined_input))
        combined_input = F.relu(self.fc5(combined_input))
        
        return torch.tanh(combined_input)
'''
class LearnerConvNetSSC(nn.Module):
    
    
    The learner that has one or more camera inputs:
        SSC - Sensor/action Sensor/action to Camera
    The learners recieves: 
    2 input action/senssor, real number between 1 and -1
    1 output 600x800 rgb array 
    
    
    
    self.__init__(self, D_in, D_fc, )
'''