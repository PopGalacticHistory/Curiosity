# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 18:39:35 2019

@author: or_ra
"""

import numpy as np
import random

def ChooseActionNoBrain(observation):
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
    if random.randrange(0,100)/100 > 0.5:
        f = 0.0
    if random.randrange(0,100)/100 >0.5:
        y = 0.0
    action = [f, y]
    return action
    