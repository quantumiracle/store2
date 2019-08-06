"""
pure vector observation based learning: position of tactip and target
task: tactip following the cylinder to reach the ball target
use 382 pins
"""

import tensorflow as tf
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten
import numpy as np
import matplotlib.pyplot as plt
import gym, threading, queue
from gym_unity.envs import UnityEnv
import argparse
from PIL import Image
from deform_visualize import plot_list_new
import pickle

data_file=open('data.pickle', "wb")
label_file=open('label.pickle', "wb")

def plot(s):
    x=s[7::3]
    z=s[9::3]
    plot_list_new(x,z)

def state_process(s):
    ps=np.concatenate((s[7::3],s[9::3]))
    return ps

if __name__ == '__main__':
    training_episodes = 100
    episode_length = 150
    obs_dim = 182  # total 280: 0 object index, 1-3 rotation value, 4-6 average contact point position, 7-279 pins positions
    state_dim = 6
    env_name = "./tac_touch_fixed"  # Name of the Unity environment binary to launch
    env = UnityEnv(env_name, worker_id=np.random.randint(0,10), use_visual=False, use_both=True)

    batch_s = []
    batch_label = []
    for eps in range(training_episodes):
        print(eps)
        s,info = env.reset()
        s0=np.array(s[7:])
        for step in range(episode_length):
            # plot(s)
            if step >0 and np.mean(np.abs(np.array(s[7:])-s0))>0.6 and s[4]+s[5]+s[6]!=0:  # set a threshold to extract deformation frames
                # print(np.mean(np.abs(np.array(s[7:])-s0)))
                # print(s[:7])
                batch_s.append(state_process(s))
                batch_label.append(np.concatenate((s[1:4]/30., s[4:7])))  # normalize the rotation range by 30, to get [-1,1]

            s_, r, done, info= env.step([0])
            s=s_
            
        print(np.array(batch_label).shape)

    pickle.dump(batch_s, data_file)
    pickle.dump(batch_label, label_file)   
