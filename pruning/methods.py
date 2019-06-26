import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pruning.utils import prune_rate, arg_nonzero_min


def neuron_prune(model, pruning_perc = 10, threshold=None):
    '''
    Prune a neuron, i.e. all contributing weights to a spacific neuron will be 
    zeroed out. Be aware that this works only with fully connected layers named
    fc1, fc2, and so on. 
    '''   
    viable = True
    all_weights = []
    for p in model.parameters():
        if len(p.data.size()) != 1:
            all_weights += list(p.cpu().data.abs().numpy().flatten())
    if threshold is None:
        threshold = np.percentile(np.array(all_weights), pruning_perc)
    
    #calculating the sum of the neuron input (row) and output (column) to be 
    #able to evaluate the strength of the neuron for prunning.
    
    row_sum = {}
    column_sum = {}
    for name, p in model.named_parameters(): #Creates a sum for each raw and column
        if p.dim() == 2: #the FC parts are two dimentional (conv are four)
            
            row_sum[name] = torch.abs(torch.sum(p, dim=1))
            column_sum[name] = torch.abs(torch.sum(p, dim=0))
    '''        
    for name, p in model.named_parameters(): ##Creates a sum for each column
        if p.dim()>1:
            column_sum[name] = torch.abs(torch.sum(p, dim=0))
    '''
    prunned_neurons_dict = {}
    prunned_neurons = [] #a place to store the index of prunnes neurons        
    for i in range((len(row_sum)-1)):
        neuron_strength = row_sum['fc' + str(i+1) + '.weight'] + column_sum['fc' + str(i+2) + '.weight']
        prunned_neurons_dict[i+1] = 0 
        
        for j in range(len(neuron_strength)):
            if neuron_strength[j] < threshold:
                #print(neuron_strength[j])
                #if neuron_strength[j] == 0:
                #    break
                #print('prunning layer:', i+1, 'neuron:', j)
                prunned_neurons.append([i+1,j]) #here i is the layer of the row (input) 
                                            # and j is index of the row/column. So 
                                            #[1,2] means the second neuron of the first hidden layer, 
                                            #so we are zeroing the second row of the first weights 
                                            #and the second column of the second weights.
                prunned_neurons_dict[i+1] += 1 #counts the number of prunned neurons per layer
    
    count = 1            
    for p in model.parameters():
        if count == len(row_sum) - 1:
            continue
        else:
            if p.dim() == 2:
                if max(p.size()) == prunned_neurons_dict[count]:
                    viable = False
                count += 1
        
                                                
    # generate mask
    masks = {}
    #creating a mask of ones the size of the weight matrixs
    for name, p in model.named_parameters():
        if p.dim() > 1:
            masks[name] = torch.ones(p.size())
    #zeroing out the appropiate row and column     
    if len(prunned_neurons) > 1:
        for neuron in prunned_neurons:
            masks['fc' + str(neuron[0]) + '.weight'][neuron[1], :] = 0
            masks['fc' + str(neuron[0] + 1) + '.weight'][:, neuron[1]] = 0
        
    
    return masks, prunned_neurons, viable 

def weight_prune(model, pruning_perc):
    '''
    Prune pruning_perc% weights globally (not layer-wise)
    arXiv: 1606.09274
    '''    
    all_weights = []
    for p in model.parameters():
        if len(p.data.size()) != 1:
            all_weights += list(p.cpu().data.abs().numpy().flatten())
    threshold = np.percentile(np.array(all_weights), pruning_perc)

    # generate mask
    masks = []
    for p in model.parameters():
        if len(p.data.size()) != 1:
            pruned_inds = p.data.abs() > threshold
            masks.append(pruned_inds.float())
    return masks

def prune_one_filter(model, masks):
    '''
    Pruning one least ``important'' feature map by the scaled l2norm of 
    kernel weights
    arXiv:1611.06440
    '''
    NO_MASKS = False
    # construct masks if there is not yet
    if not masks:
        masks = []
        NO_MASKS = True

    values = []
    for p in model.parameters():

        if len(p.data.size()) == 4: # nasty way of selecting conv layer
            p_np = p.data.cpu().numpy()

            # construct masks if there is not
            if NO_MASKS:
                masks.append(np.ones(p_np.shape).astype('float32'))

            # find the scaled l2 norm for each filter this layer
            value_this_layer = np.square(p_np).sum(axis=1).sum(axis=1)\
                .sum(axis=1)/(p_np.shape[1]*p_np.shape[2]*p_np.shape[3])
            # normalization (important)
            value_this_layer = value_this_layer / \
                np.sqrt(np.square(value_this_layer).sum())
            min_value, min_ind = arg_nonzero_min(list(value_this_layer))
            values.append([min_value, min_ind])

    assert len(masks) == len(values), "something wrong here"

    values = np.array(values)

    # set mask corresponding to the filter to prune
    to_prune_layer_ind = np.argmin(values[:, 0])
    to_prune_filter_ind = int(values[to_prune_layer_ind, 1])
    masks[to_prune_layer_ind][to_prune_filter_ind] = 0.

    print('Prune filter #{} in layer #{}'.format(
        to_prune_filter_ind, 
        to_prune_layer_ind))

    return masks


def filter_prune(model, pruning_perc):
    '''
    Prune filters one by one until reach pruning_perc
    (not iterative pruning)
    '''
    masks = []
    current_pruning_perc = 0.

    while current_pruning_perc < pruning_perc:
        masks = prune_one_filter(model, masks)
        model.set_masks(masks)
        current_pruning_perc = prune_rate(model, verbose=False)
        print('{:.2f} pruned'.format(current_pruning_perc))

    return masks
