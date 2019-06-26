# -*- coding: utf-8 -*-
"""

Lets see it all run!! 

"""
import gym
import gym_autoRobo
import numpy as np
import matplotlib.pyplot as plt

from LearningSteps import LearningStep
import utilities 

learn = LearningStep()
learn.flatten_image = False
learn.prunning_treshold = 0.5

print(vars(learn))
learn.initialize_nets(all_nets = False, selected_nets = [[["camera t-1", "camera action t"], "camera t"],
                                                         [["camera angle t-1", "arm action t"], "camera angle t"]] )

env = gym.make('autoRobo-v0')
env.reset()
observation = env.reset()
epochs = 150

for t in range(epochs):
    #print(t)
    action = utilities.ChooseActionNoBrain(observation)
    observation, reward, done, info = env.step(action) #observation includes
    #camera angles, arm angles, rgb image, actions - all in time t
    #the order is as follows:
    #[arm angle, camera angle, arm action, camera action, rgb image]
    learn.get_data(observation, t) 
    if t>1:
        learn.onlineStep()
    
env.close()

b = 1
plt.subplots_adjust(left=5, bottom=5, right=6.5, top=6.5, wspace=None, hspace=1)
for a in learn.loss_dict:
    ax = plt.subplot(len(learn.loss_dict), 1, b )
    b += 1
    plt.plot(learn.loss_dict[a])
    ax.set_title(a)
    




 
    