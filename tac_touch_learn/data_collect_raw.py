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
# from deform_visualize import plot_list_new
import pickle

data_file=open('raw_data.pickle', "wb")

if __name__ == '__main__':
    training_episodes = 1000
    episode_length = 150
    # env_name = "./tac_touch_fixed"  # Name of the Unity environment binary to launch
    # env_name = "./tac_touch_random"  # random range 0.5
    # env_name = "./tac_touch_random283"  # random range 0.5
    env_name = "./tac_touch_random283_2"  # random range 0.2
    # env_name = "./tac_touch_fixed283"  # random range 0.0

    env = UnityEnv(env_name, worker_id=np.random.randint(0,10), use_visual=False, use_both=True)

    batch_s = []
    cnt=0
    for eps in range(training_episodes):
        print(eps)
        s,info = env.reset()
        s0=np.array(s)
        for step in range(episode_length):
            # plot(s)
            # print(np.mean(np.abs(np.array(s[7:])-s0)))  # choose 0.03
            if step >0 and np.mean(np.abs(np.array(s[10:])-s0[10:]))>0.03 and s[4]+s[5]+s[6]!=0:  # set a threshold to extract deformation frames
                batch_s.append(s)     # dim of s total 280 (select 182 as obs): 0 object index, 1-3 rotation value, 4-6 average contact point position, 7-279 pins positions
                cnt+=1

            s_, r, done, info= env.step([0])
            s=s_
    print('total number of samples: ', cnt)
    pickle.dump(batch_s, data_file)
