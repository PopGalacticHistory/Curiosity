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
import numpy as np
from skimage.transform import rescale
from skimage.util import crop

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

class LearningStep(object):
    
    def __init__(self):
        super(self.__class__, self).__init__()
        self.learning_rate_FC = 0.1
        self.learning_rate_FC_image = 0.001
        self.learning_rate_ConvNet = 0.1
        self.optimizerFC = "Adam"
        self.optimizerConvNet = "Adam"
        self.criterionConvNet = nn.CrossEntropyLoss()
        self.criterionFC = nn.MSELoss()
        self.prunning_treshold = 0.01
        self.flatten_image = False
        self.binary_action = True

    
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
        self.net_dict = {}
        self.loss_dict = {}
        self.viable_nets = {}
        
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
        
        if self.flatten_image == True:
            CCS_net = Learner_Nets.LearnerFCNets2().state_dict() 
            CSC_net = Learner_Nets.LearnerFCNets2().state_dict()  
            CSS_net = Learner_Nets.LearnerFCNets2().state_dict()
        else:
            CCS_net = Learner_Nets.LearnerConvNetCCSOneInput().state_dict()
            CSC_net = Learner_Nets.LearnerConvNetCSCOneInput().state_dict()
            CSS_net =2# Learner_Nets.LearnerConvNetCSS().state_dict()
        SSC_net = None 
        SSS_net = Learner_Nets.LearnerFCNets2().state_dict()
        

        for name in nets_name:
            if "camera t-1" in name[0]:
                if "camera t" in name[0]:
                    self.net_dict[name[0][0],name[0][1], name[1], 'CCS'] = copy.deepcopy(CCS_net)
                    self.loss_dict[name[0][0],name[0][1], name[1], 'CCS'] = []
                    self.loss_dict[name[0][0],name[0][1], name[1], 'CCS', 'mean'] = []
                    self.viable_nets[name[0][0],name[0][1], name[1], 'CCS'] = True
                elif "camera t" in name:
                    self.net_dict[name[0][0],name[0][1], name[1], 'CSC'] = copy.deepcopy(CSC_net)
                    self.loss_dict[name[0][0],name[0][1], name[1], 'CSC'] = []
                    self.loss_dict[name[0][0],name[0][1], name[1], 'CSC', 'mean'] = []
                    self.viable_nets[name[0][0],name[0][1], name[1], 'CSC'] = True
                else:
                    self.net_dict[name[0][0],name[0][1], name[1], 'CSS'] = copy.deepcopy(CSS_net)
                    self.loss_dict[name[0][0],name[0][1], name[1], 'CSS'] = []
                    self.loss_dict[name[0][0],name[0][1], name[1], 'CSS', 'mean'] = []
                    self.viable_nets[name[0][0],name[0][1], name[1], 'CSS'] = True
            elif "camera t" in name[0]:
                if "camera t-1" in name:
                    self.net_dict[name[0][0],name[0][1], name[1], 'CSC'] = copy.deepcopy(CSC_net)
                    self.loss_dict[name[0][0],name[0][1], name[1], 'CSC'] = []
                    self.loss_dict[name[0][0],name[0][1], name[1], 'CSC', 'mean'] = []
                    self.viable_nets[name[0][0],name[0][1], name[1], 'CSC'] = True
                else:
                    self.net_dict[name[0][0],name[0][1], name[1], 'CSS'] = copy.deepcopy(CSS_net)
                    self.loss_dict[name[0][0],name[0][1], name[1], 'CSS'] = []
                    self.loss_dict[name[0][0],name[0][1], name[1], 'CSS', 'mean'] = []
                    self.viable_nets[name[0][0],name[0][1], name[1], 'CSS'] = True
            else:
                if "camera t" in name:
                    self.net_dict[name[0][0],name[0][1], name[1], 'SSC'] = copy.deepcopy(SSC_net)
                    self.loss_dict[name[0][0],name[0][1], name[1], 'SSC'] = []
                    self.loss_dict[name[0][0],name[0][1], name[1], 'SSC', 'mean'] = []
                    self.viable_nets[name[0][0],name[0][1], name[1], 'SSC'] = True
                elif "camera t-1" in name:
                    self.net_dict[name[0][0],name[0][1], name[1], 'SSC'] = copy.deepcopy(SSC_net)
                    self.loss_dict[name[0][0],name[0][1], name[1], 'SSC'] = []
                    self.loss_dict[name[0][0],name[0][1], name[1], 'SSC', 'mean'] = []
                    self.viable_nets[name[0][0],name[0][1], name[1], 'SSC'] = True
                else:
                    self.net_dict[name[0][0],name[0][1], name[1], 'SSS'] = copy.deepcopy(SSS_net)
                    self.loss_dict[name[0][0],name[0][1], name[1], 'SSS'] = []
                    self.loss_dict[name[0][0],name[0][1], name[1], 'SSS', 'mean'] = []
                    self.viable_nets[name[0][0],name[0][1], name[1], 'SSS'] = True
                                                                

    
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
            
    def onlineStep(self):
        for learner in  self.net_dict:
            if self.viable_nets[learner]:
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
                        masks, prunned_neurons, vaible = neuron_prune(net, pruning_perc=0.1, threshold=self.prunning_treshold)
                        net.set_masks(masks)
                        self.viable_nets[learner] = vaible
                        if not vaible:
                            print('prunned the following network: ', learner)
                    self.net_dict[learner] = copy.deepcopy(net.state_dict())
                    self.loss_dict[learner].append(loss.item())  
                    self.loss_dict[learner + ('mean',)].append(np.mean(self.loss_dict[learner]))
                elif learner[3] == 'SSC':
                    continue
                elif learner[3] == 'CSS':
                    continue
                elif learner[3] == 'CCS':
                    if self.flatten_image:
                        net = Learner_Nets.LearnerFCNets2()
                        net.load_state_dict(self.net_dict[learner])
                        optimizer = optim.Adam(net.parameters(), 
                                               lr = self.learning_rate_FC_image)
                        if learner[2] in ["camera action t", "arm action t"]:
                            binary_action = self.binary_action
                        else:
                            binary_action = False
                        fc_camera_input, target = utilities.create_pixel_vecCC(self.learning_dataset['camera t-1'],
                                                                               self.learning_dataset['camera t'],
                                                                               self.learning_dataset[learner[2]],
                                                                               rescale_image = True, 
                                                                               binary_action=binary_action)
                    
                        #if target != 0:
                        #    target = torch.tensor([1])
                        #else:
                        #    target = torch.tensor([0])
                        learn_data = torch.utils.data.TensorDataset(fc_camera_input, target)
                        learn_loader = torch.utils.data.DataLoader(learn_data, batch_size=30)
                        optimizer = optim.Adam(net.parameters(), lr=self.learning_rate_FC)
                        loss_vec = []
                        for train, label in learn_loader:
                            label = label.squeeze(0)
                            optimizer.zero_grad() 
                            output = net(train)  
                            loss = self.criterionFC(output, label)
                            loss.backward()
                            optimizer.step()
                            if self.time>100:
                                masks, prunned_neurons, vaible = neuron_prune(net, pruning_perc=0.1, threshold=self.prunning_treshold)
                                net.set_masks(masks)
                                self.viable_nets[learner] = vaible
                                if not vaible:
                                    print('prunned the following network: ', learner)
                            self.net_dict[learner] = copy.deepcopy(net.state_dict())
                            loss_vec.append(loss.item())
                        self.loss_dict[learner].append(np.mean(loss_vec))
                        self.loss_dict[learner + ('mean',)].append(np.mean(self.loss_dict[learner]))
                    
                    else:
                        net = Learner_Nets.LearnerConvNetCCSOneInput()
                        net.load_state_dict(self.net_dict[learner])
                        input_ = torch.empty(3, 240*2, 320)
                        input_[:, :240, :] = self.learning_dataset['camera t-1'].permute(2,0,1)
                        input_[:, 240:, :] = self.learning_dataset['camera t'].permute(2,0,1)
                        input_ = input_.unsqueeze(0)
                        target = self.learning_dataset[learner[2]]
                        if learner[2] in ["camera action t", "arm action t"]:
                            binary_action = self.binary_action
                        else:
                            binary_action = False
                        if binary_action:
                            if target != 0:
                                target = torch.tensor([1])
                            else:
                                target = torch.tensor([0])
                        learn_data = torch.utils.data.TensorDataset(input_, target.long())
                        learn_loader = torch.utils.data.DataLoader(learn_data, batch_size=1)
                        optimizer = optim.Adam(net.parameters(), lr=self.learning_rate_ConvNet)
                        for train, label in learn_loader:
                            #label = label.squeeze(0)
                            optimizer.zero_grad() 
                            output = net(train)  
                            loss = self.criterionConvNet(output, label)
                            loss.backward()
                            optimizer.step()
                            if self.time>100:
                                masks, prunned_neurons, vaible = neuron_prune(net, pruning_perc=0.1, threshold=self.prunning_treshold)
                                net.set_masks(masks)
                                self.viable_nets[learner] = vaible
                                if not vaible:
                                    print('prunned the following network: ', learner)
                            self.net_dict[learner] = copy.deepcopy(net.state_dict())
                            self.loss_dict[learner].append(loss.item())  
                            self.loss_dict[learner + ('mean',)].append(np.mean(self.loss_dict[learner]))
                elif learner[3] == 'CSC':
                    if self.flatten_image:
                        net = Learner_Nets.LearnerFCNets2()
                        net.load_state_dict(self.net_dict[learner])
                        optimizer = optim.Adam(net.parameters(), 
                                               lr = self.learning_rate_FC_image)
                        binary_action = False
                        fc_camera_input, target = utilities.create_pixel_vecCS(self.learning_dataset[learner[0]],
                                                                               self.learning_dataset[learner[1]], 
                                                                               self.learning_dataset[learner[2]],
                                                                               rescale_image = True, 
                                                                               binary_action = binary_action)
                        learn_data = torch.utils.data.TensorDataset(fc_camera_input, target.float())
                        learn_loader = torch.utils.data.DataLoader(learn_data, batch_size=30)
                        loss_vec = []
                        for train, label in learn_loader:
                            label = label.squeeze(0)
                            optimizer.zero_grad() 
                            output = net(train)  
                            loss = self.criterionFC(output, label)
                            loss.backward()
                            optimizer.step()
                            if self.time>100:
                                masks, prunned_neurons, vaible = neuron_prune(net, pruning_perc=0.1, threshold=self.prunning_treshold)
                                net.set_masks(masks)
                                self.viable_nets[learner] = vaible
                                if not vaible:
                                    print('prunned the following network: ', learner)
                            self.net_dict[learner] = copy.deepcopy(net.state_dict())
                            loss_vec.append(loss.item())
                        self.loss_dict[learner].append(np.mean(loss_vec))
                        self.loss_dict[learner + ('mean',)].append(np.mean(self.loss_dict[learner]))
                    
                    else:
                        net = Learner_Nets.LearnerConvNetCSCOneInput()
                        net.load_state_dict(self.net_dict[learner])
                        rescaled_image = rescale(self.learning_dataset[learner[0]].data.numpy(), 1.0/2.0, anti_aliasing=True, multichannel=True)
                        rescaled_image = crop(rescaled_image, ((0,0),(20,20), (0,0)))
                        input_ = torch.empty(4, 120, 120)
                        input_[:3, :, :] = torch.from_numpy(rescaled_image).permute(2,0,1)
                        #print(self.learning_dataset[learner[1]])
                        input_[3, :, :] = self.learning_dataset[learner[1]]
                        input_ = input_.unsqueeze(0)
                        target = rescale(self.learning_dataset[learner[2]].data.numpy(), 
                                         1.0/2.0, anti_aliasing=True, multichannel=True)
                        target = crop(target, ((0,0),(20,20), (0,0)))
                        target = torch.from_numpy(target).permute(2,0,1).float()
                        #learn_data = torch.utils.data.TensorDataset(input_, target)
                        #learn_loader = torch.utils.data.DataLoader(learn_data, batch_size=1)
                        optimizer = optim.Adam(net.parameters(), lr=self.learning_rate_ConvNet)
                        #for train, label in learn_loader:
                        #target = target.squeeze(0)
                        optimizer.zero_grad() 
                        output = net(input_)  
                        criterion = nn.MSELoss(reduce = False)
                        #output = output.view(120*120*3)
                        target = target.view(120*120*3)
                        loss = criterion(output, target)
                        loss.mean().backward()
                        
                        optimizer.step()
                        if self.time>100:
                            masks, prunned_neurons, vaible = neuron_prune(net, pruning_perc=0.1, threshold=self.prunning_treshold)
                            net.set_masks(masks)
                            self.viable_nets[learner] = vaible
                            if not vaible:
                                print('prunned the following network: ', learner)
                        self.net_dict[learner] = copy.deepcopy(net.state_dict())
                        self.loss_dict[learner].append(loss.mean().item())
                        self.loss_dict[learner + ('mean',)].append(np.mean(self.loss_dict[learner]))
                
        
        
            
    
            
        
        
        
    
        
        
        
        
        
        
        