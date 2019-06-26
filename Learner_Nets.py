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
        self.fc1 = nn.Linear(2, 4)
        self.bn1 = nn.BatchNorm1d(4)
        self.fc2 = nn.Linear(4, 4)
        self.bn2 = nn.BatchNorm1d(4)
        self.fc3 = nn.Linear(4, 4)
        self.fc4 = nn.Linear(4, 1)
        
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        return torch.tanh(self.fc4(x))
    
    def set_masks(self, masks):
        self.fc1.weight.data = torch.mul(self.fc1.weight.data, masks['fc1.weight'])
        self.fc2.weight.data = torch.mul(self.fc2.weight.data, masks['fc2.weight'])
        self.fc3.weight.data = torch.mul(self.fc3.weight.data, masks['fc3.weight'])
        self.fc4.weight.data = torch.mul(self.fc4.weight.data, masks['fc4.weight'])
        
class LearnerConvNetCCSOneInput(nn.Module):
    def __init__(self):
        super(LearnerConvNetCCSOneInput, self).__init__()
        '''
        The learner that has one or more camera inputs:
            CCS - Camera Camera to Sensor/action
            The learners recieves: 
                1 input from camera which is a combination of two 240X320X3
                images of time t and t-1. To learn:
                1 output action/senssor
        While using the data loader of pytorch it is easier to use One Input 
        format. 
        '''
    
        self.conv1 = nn.Conv2d(3, 3, kernel_size = 1, stride = 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(3, 5, kernel_size = 3, stride = 2)
        self.conv3 = nn.Conv2d(5, 10, kernel_size = 5, stride = 2)
        self.fc1 = nn.Linear(20*6*9, 260)
        self.fc2 = nn.Linear(260, 125)
        self.fc3 = nn.Linear(125, 80)
        self.fc4 = nn.Linear(80, 16)
        self.fc5 = nn.Linear(16, 2)
       
    def forward(self, x):
        image1 = x[:, :, :240, :]
        image2 = x[:, :, 240:, :] 
        image1 = self.pool(F.relu(self.conv1(image1))) #after it is 120X160X10
        image2 = self.pool(F.relu(self.conv1(image2))) #after it is 120X160X10
        image1 = self.pool(F.relu(self.conv2(image1))) #29X39X10
        image2 = self.pool(F.relu(self.conv2(image2))) #29X39X10
        image1 = self.pool(F.relu(self.conv3(image1))) #6X9X10
        image2 = self.pool(F.relu(self.conv3(image2))) #6X9X10
        image1 = image1.view(image1.size(0), 10*6*9)
        image2 = image2.view(image2.size(0), 10*6*9)
        combined_input = torch.cat((image1, image2), -1)
        combined_input = F.relu(self.fc1(combined_input))
        combined_input = F.relu(self.fc2(combined_input))
        combined_input = F.relu(self.fc3(combined_input))
        combined_input = F.relu(self.fc4(combined_input))
        combined_input = self.fc5(combined_input)
        
        return combined_input
    
    def set_masks(self, masks):
        self.fc1.weight.data = torch.mul(self.fc1.weight.data, masks['fc1.weight'])
        self.fc2.weight.data = torch.mul(self.fc2.weight.data, masks['fc2.weight'])
        self.fc3.weight.data = torch.mul(self.fc3.weight.data, masks['fc3.weight'])
        self.fc4.weight.data = torch.mul(self.fc4.weight.data, masks['fc4.weight'])
        self.fc5.weight.data = torch.mul(self.fc5.weight.data, masks['fc5.weight'])
        
class LearnerConvNetCSCOneInput(nn.Module):
    
    def __init__(self):
        super(LearnerConvNetCSCOneInput, self).__init__()
        '''
        The learner that has one or more camera inputs:
        CSS - Camera Sensor/action to Sensor/action
        The learners recieves: 
        1 input action/senssor, real number between 1 and -1
        1 input from camera, 120x120 rgb array 
        1 output iamge 120x120
        
        While using the data loader of pytorch it is easier to use One Input 
        format. 
       '''
        #input (4,120,120)
        self.conv1 = nn.Conv2d(4, 96, kernel_size = 5, stride = 1)
        self.batchNorm1 = nn.BatchNorm2d(96, momentum=1)
        self.pool = nn.MaxPool2d(2, 2,return_indices=True)
        #input (96, 58, 58)
        self.conv2 = nn.Conv2d(96, 64, kernel_size = 5, stride = 1)
        self.batchNorm2 = nn.BatchNorm2d(64, momentum=1)
        #input (64, 27, 27)
        self.conv3 = nn.Conv2d(64, 32, kernel_size = 5, stride = 1)
        self.batchNorm3 = nn.BatchNorm2d(32, momentum=1)
        # input (32, 11, 11)
        self.fc1 = nn.Linear(11*11*32, 1900)
        self.fc2 = nn.Linear(1900, 700)
        self.fc3 = nn.Linear(700, 1900)
        self.fc4 = nn.Linear(1900, 11*11*32)
        self.conv4 = nn.ConvTranspose2d(32, 64, kernel_size = 5, stride = 1)
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.conv5 = nn.ConvTranspose2d(64, 96, kernel_size = 5, stride = 1)
        self.conv6 = nn.ConvTranspose2d(96, 3, kernel_size = 5, stride = 1)

    def forward(self, x):
        combined = x
        combined, indices1 = self.pool(F.relu(self.batchNorm1(self.conv1(combined))))
        combined, indices2 = self.pool(F.relu(self.batchNorm2(self.conv2(combined))))
        combined, indices3 = self.pool(F.relu(self.batchNorm3(self.conv3(combined))))
        combined = combined.view(11*11*32)
        combined = F.relu(self.fc1(combined))
        combined = F.relu(self.fc2(combined))
        combined = F.relu(self.fc3(combined))
        combined = F.relu(self.fc4(combined))
        combined = combined.view([32, 11, 11]).unsqueeze(0)
        combined = F.relu(self.batchNorm2(self.conv4(self.unpool(combined, indices3, [1,32,23,23]))))
        combined = F.relu(self.batchNorm1(self.conv5(self.unpool(combined, indices2, [1,64, 54,54]))))
        combined = F.relu(self.conv6(self.unpool(combined, indices1, [1,96,116,116])))
        combined = combined.squeeze(0)
        combined = combined.view(120*120*3)
        
        return combined 
    
    def set_masks(self, masks):
        self.fc1.weight.data = torch.mul(self.fc1.weight.data, masks['fc1.weight'])
        self.fc2.weight.data = torch.mul(self.fc2.weight.data, masks['fc2.weight'])
        self.fc3.weight.data = torch.mul(self.fc3.weight.data, masks['fc3.weight'])
        self.fc4.weight.data = torch.mul(self.fc4.weight.data, masks['fc4.weight'])

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
                2 input from camera, 240X320X3 
                1 output action/senssor
        '''
    
        self.conv1 = nn.Conv2d(3, 10, kernel_size = 1, stride = 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(10, 10, kernel_size = 3, stride = 2)
        self.conv3 = nn.Conv2d(10, 10, kernel_size = 5, stride = 2)
        self.fc1 = nn.Linear(20*6*9, 260)
        self.fc2 = nn.Linear(260, 125)
        self.fc3 = nn.Linear(125, 80)
        self.fc4 = nn.Linear(80, 16)
        self.fc5 = nn.Linear(16, 1)
       
    def forward(self, x, y):
        image1 = x
        image2 = y 
        image1 = self.pool(F.relu(self.conv1(image1))) #after it is 120X160X10
        image2 = self.pool(F.relu(self.conv1(image2))) #after it is 120X160X10
        image1 = self.pool(F.relu(self.conv2(image1))) #29X39X10
        image2 = self.pool(F.relu(self.conv2(image2))) #29X39X10
        image1 = self.pool(F.relu(self.conv3(image1))) #6X9X10
        image2 = self.pool(F.relu(self.conv3(image2))) #6X9X10
        image1 = image1.view(image1.size(0), 10*6*9)
        image2 = image2.view(image2.size(0), 10*6*9)
        combined_input = torch.cat((image1, image2), -1)
        combined_input = F.relu(self.fc1(combined_input))
        combined_input = F.relu(self.fc2(combined_input))
        combined_input = F.relu(self.fc3(combined_input))
        combined_input = F.relu(self.fc4(combined_input))
        combined_input = self.fc5(combined_input)
        
        return combined_input
    




'''
class LearnerConvNetSSC(nn.Module):
    
    
    The learner that has one or more camera inputs:
        SSC - Sensor/action Sensor/action to Camera
    The learners recieves: 
    2 input action/senssor, real number between 1 and -1
    1 output 600x800 rgb array 
    
    
    
    self.__init__(self, D_in, D_fc, )
'''