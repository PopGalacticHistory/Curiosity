# -*- coding: utf-8 -*-
"""

Creating a Class that will do a step for the online learning. 
This class will recieve the raw input from the program, divide the learners
and run a forward step using the Learner_Nets. 


"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import copy

import utilities 
from itertools import combinations
import Learner_Nets
from pruning.methods import neuron_prune

'''
information about the enviorment:
observation includes camera angles, arm angles, rgb image, actions - all in 
time t the order is as follows:
[arm angle, camera angle, arm action, camera action, rgb image]
    
'''

class onlineLearningStep(object):
    
    def __init__(self):
        super(self.__class__, self).__init__()
        self.learning_rate_FC = 0.1
        self.learning_rate_ConvNet = 0.1
        self.optimizerFC = "Adam"
        self.optimizerConvNet = "Adam"
        self.criterionConvNet = nn.CrossEntropyLoss()
        self.criterionFC = nn.MSELoss()
        self.prunning_treshold = 0.01
        self.flatten_image = False
        self.net_dict = {}
        self.loss_dict = {}
    
    def initialize_nets(self, all_nets = True, selected_nets = None):
        '''
        This function will initialize the networks for learning. 
        
        Arguments:
            all_nets     - whether to initialize all the networks together, or
                           to only initialize a number of selected nets.
                           Set to be True as default
            selected_nets - if all_nets=False than a list of selected networks 
                           should be entered in the following format:
                               [['input 1', 'input 2'], 'target']
                           for example:
                               [['camera t', 'camera angle t'], 'camera t-1']   
                           Set to be None as default. 
        
        
        
        A function that takes in the base models of the learners and creates a dict 
        holding relevent nets for relevent learners. Fully connected networks for 
        the action/sensors correlations and convelution networks for the correleations
        with one or more image inputs. 
        
        Return:
            net_dict - a python dictionery holding the state_dict of the initialized networks
            loss_dict - a python dictionery holding spaces for the loss of each
                        network. To be used as input for the RL network. 
    
        '''
        
        input_list = ["camera t-1", "camera t", "camera angle t-1", "camera angle t", 
                      "arm angle t-1", "arm angle t", "camera action t", "arm action t"]

            
        comb_list = []
        
        for a, b in combinations(input_list, 2):
            comb_list.append([a, b])
        
        if all_nets:
            nets_name = []
            for name in input_list:
                for comb in comb_list:
                    if name not in comb:
                        nets_name.append([comb, name])
        else:
            nets_name = selected_nets
        
        if self.flatten_image:
            CCS_net = Learner_Nets.LearnerFCNets2().state_dict() 
            CSC_net = Learner_Nets.LearnerFCNets2().state_dict()  
            CSS_net = Learner_Nets.LearnerFCNets2().state_dict()
        else:
            CCS_net = Learner_Nets.LearnerConvNetCCSOneInput().state_dict()
            CSC_net = 1#Learner_Nets.LearnerConvNetCSC().state_dict()
            CSS_net =2# Learner_Nets.LearnerConvNetCSS().state_dict()
        SSC_net = None 
        SSS_net = Learner_Nets.LearnerFCNets2().state_dict()
        

        for name in nets_name:
            if "camera t-1" in name[0]:
                if "camera t" in name[0]:
                    self.net_dict[name[0][0],name[0][1], name[1], 'CCS'] = copy.deepcopy(CCS_net)
                    self.loss_dict[name[0][0],name[0][1], name[1], 'CCS'] = []
                elif "camera t" in name:
                    self.net_dict[name[0][0],name[0][1], name[1], 'CSC'] = copy.deepcopy(CSC_net)
                    self.loss_dict[name[0][0],name[0][1], name[1], 'CSC'] = []
                else:
                    self.net_dict[name[0][0],name[0][1], name[1], 'CSS'] = copy.deepcopy(CSS_net)
                    self.loss_dict[name[0][0],name[0][1], name[1], 'CSS'] = []
            elif "camera t" in name[0]:
                if "camera t-1" in name:
                    self.net_dict[name[0][0],name[0][1], name[1], 'CSC'] = copy.deepcopy(CSC_net)
                    self.loss_dict[name[0][0],name[0][1], name[1], 'CSC'] = []
                else:
                    self.net_dict[name[0][0],name[0][1], name[1], 'CSS'] = copy.deepcopy(CSS_net)
                    self.loss_dict[name[0][0],name[0][1], name[1], 'CSS'] = []
            else:
                if "camera t" in name:
                    self.net_dict[name[0][0],name[0][1], name[1], 'SSC'] = copy.deepcopy(SSC_net)
                    self.loss_dict[name[0][0],name[0][1], name[1], 'SSC'] = []
                elif "camera t-1" in name:
                    self.net_dict[name[0][0],name[0][1], name[1], 'SSC'] = copy.deepcopy(SSC_net)
                    self.loss_dict[name[0][0],name[0][1], name[1], 'SSC'] = []
                else:
                    self.net_dict[name[0][0],name[0][1], name[1], 'SSS'] = copy.deepcopy(SSS_net)
                    self.loss_dict[name[0][0],name[0][1], name[1], 'SSS'] = []
                                                                

    
    def get_data(self, observation, time):
        
        '''
        Arguments:
            observation - the observaion from the autoRobo enviorment, orgenized
                         includes camera angles, arm angles, rgb image, actions 
                         - all in time t. the order is as follows:
                         [arm angle, camera angle, arm action, camera action, rgb image]
            time        - the number of the epoch, to know if an observation
                            has been registered allready or not. 
        Gets the observation and orgenize it to a list of all the observations
        and a dictionery that is the input of the network. 
        '''
        
        self.time = time
        if time>0:
            self.AllImage += [observation[1]]
            self.AllAngles += [observation[0][:2]]
            self.AllActions += [observation[0][2:]]
            self.learning_dataset, _ = utilities.divide_learners(self.AllActions, 
                                                          self.AllAngles, 
                                                          self.AllImage, 
                                                          time = time)
        elif time==0:
            self.AllImage = [observation[1]]
            self.AllAngles = [observation[0][:2]]
            self.AllActions = [observation[0][2:]]
            
    def step(self):
        for learner in  self.net_dict:
            if learner[3] == 'SSS':
                net = Learner_Nets.LearnerFCNets2()
                net.load_state_dict(self.net_dict[learner])
                optimizer = optim.Adam(net.parameters(), 
                                      lr = self.learning_rate_FC)
                input_ = torch.tensor([self.learning_dataset[learner[0]], self.learning_dataset[learner[1]]])
                target = self.learning_dataset[learner[2]]
                optimizer.zero_grad() 
                output = net(input_)  
                #print(output.item(), y_target)
                loss = self.criterionFC(output, target)
                loss.backward()
                optimizer.step()
                if self.time>100:
                    masks, prunned_neurons = neuron_prune(net, pruning_perc=0.1, threshold=self.prunning_treshold)
                    net.set_masks(masks)
                self.net_dict[learner] = copy.deepcopy(net.state_dict())
                self.loss_dict[learner].append(loss.item())  
            elif learner[3] == 'SSC':
                continue
            elif learner[3] == 'CSS':
                continue
            elif learner[3] == 'CCS':
                if self.flatten_image:
                    net = Learner_Nets.LearnerFCNets2()
                    net.load_state_dict(self.net_dict[learner])
                    optimizer = optim.Adam(net.parameters(), 
                                      lr = self.learning_rate_FC)
                    fc_camera_input = utilities.create_pixel_vecCC(self.learning_dataset['camera t-1'],
                                                                   self.learning_dataset['camera t'], 
                                                                   rescale_image = True)
                    target = self.learning_dataset[learner[2]]
                    if target != 0:
                        target = torch.tensor([1])
                    else:
                        target = torch.tensor([0])
                    learn_data = torch.utils.data.TensorDataset(fc_camera_input, target.long())
                    learn_loader = torch.utils.data.DataLoader(learn_data, batch_size=30)
                    optimizer = optim.Adam(net.parameters(), lr=self.learning_rate_FC)
                    for train, label in learn_loader:
                        label = label.squeeze(0)
                        optimizer.zero_grad() 
                        output = net(train)  
                        loss = self.criterionConvNet(output, label)
                        loss.backward()
                        optimizer.step()
                        if self.time>100:
                            masks, prunned_neurons = neuron_prune(net, pruning_perc=0.1, threshold=self.prunning_treshold)
                            net.set_masks(masks)
                        self.net_dict[learner] = copy.deepcopy(net.state_dict())
                        self.loss_dict[learner].append(loss.item())  
                    
                else:
                    net = Learner_Nets.LearnerConvNetCCSOneInput()
                    net.load_state_dict(self.net_dict[learner])
                    input_ = torch.empty(3, 240*2, 320)
                    input_[:, :240, :] = self.learning_dataset['camera t-1'].permute(2,0,1)
                    input_[:, 240:, :] = self.learning_dataset['camera t'].permute(2,0,1)
                    input_ = input_.unsqueeze(0)
                    target = self.learning_dataset[learner[2]]
                    if target != 0:
                        target = torch.tensor([1])
                    else:
                        target = torch.tensor([0])
                    learn_data = torch.utils.data.TensorDataset(input_, target.long())
                    learn_loader = torch.utils.data.DataLoader(learn_data, batch_size=1)
                    optimizer = optim.Adam(net.parameters(), lr=self.learning_rate_ConvNet)
                    for train, label in learn_loader:
                        label = label.squeeze(0)
                        optimizer.zero_grad() 
                        output = net(train)  
                        loss = self.criterionConvNet(output, label)
                        loss.backward()
                        optimizer.step()
                        if self.time>100:
                            masks, prunned_neurons = neuron_prune(net, pruning_perc=0.1, threshold=self.prunning_treshold)
                            net.set_masks(masks)
                        self.net_dict[learner] = copy.deepcopy(net.state_dict())
                        self.loss_dict[learner].append(loss.item())  
            elif learner[3] == 'CSC':
                if self.flatten_image:
                    net = Learner_Nets.LearnerFCNets2()
                    net.load_state_dict(self.net_dict[learner])
                    optimizer = optim.Adam(net.parameters(), 
                                      lr = self.learning_rate_FC)
                    
                    fc_camera_input, target = utilities.create_pixel_vecCS(self.learning_dataset[learner[0]],
                                                                   self.learning_dataset[learner[1]], 
                                                                   self.learning_dataset[learner[2]],
                                                                   rescale_image = True)
                    learn_data = torch.utils.data.TensorDataset(fc_camera_input, target.long())
                    learn_loader = torch.utils.data.DataLoader(learn_data, batch_size=30)
                    optimizer = optim.Adam(net.parameters(), lr=self.learning_rate_FC)
                    for train, label in learn_loader:
                        label = label.squeeze(0)
                        optimizer.zero_grad() 
                        output = net(train)  
                        loss = self.criterionFC(output, label)
                        loss.backward()
                        optimizer.step()
                        if self.time>100:
                            masks, prunned_neurons = neuron_prune(net, pruning_perc=0.1, threshold=self.prunning_treshold)
                            net.set_masks(masks)
                        self.net_dict[learner] = copy.deepcopy(net.state_dict())
                        self.loss_dict[learner].append(loss.item())  
                    
                else:
                    net = Learner_Nets.LearnerConvNetCSC()
                    net.load_state_dict(self.net_dict[learner])
                    input_ = torch.empty(4, 240, 320)
                    input_[:3, :, :] = self.learning_dataset[learner[0]].permute(2,0,1)
                    input_[4, :, :] = self.learning_dataset[learner[1]]
                    input_ = input_.unsqueeze(0)
                    target = self.learning_dataset[learner[2]]
                    learn_data = torch.utils.data.TensorDataset(input_, target.long())
                    learn_loader = torch.utils.data.DataLoader(learn_data, batch_size=1)
                    optimizer = optim.Adam(net.parameters(), lr=self.learning_rate_ConvNet)
                    for train, label in learn_loader:
                        label = label.squeeze(0)
                        optimizer.zero_grad() 
                        output = net(train)  
                        loss = self.criterionConvNet(output, label)
                        loss.backward()
                        optimizer.step()
                        self.net_dict[learner] = copy.deepcopy(net.state_dict())
                        self.loss_dict[learner].append(loss.item())  
                
        
        
            
    
            
        
        
        
    
        
        
        
        
        
        
        