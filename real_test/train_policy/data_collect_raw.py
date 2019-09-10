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
from td3_multiprocess_all import TD3_Trainer, ReplayBuffer

data_file=open('raw_data.pickle', "wb")

# def plot(s):
#     x=s[7::3]
#     z=s[9::3]
#     plot_list_new(x,z)

def state_process(s):
    factor=0.5674

    x0=s[3::3]
    z0=s[5::3]
    x=np.array(x0)/factor
    z=np.array(z0)/factor

    return np.transpose([x,z]).reshape(-1)  # (x,y,x,y,...)

if __name__ == '__main__':
    training_episodes = 10
    episode_length = 100

    # hyper-parameters for RL training
    max_episodes  = 5000
    max_steps   = 100
    batch_size  = 256
    explore_steps = 400  # for random action sampling in the beginning of training
    update_itr = 1
    explore_noise_scale=1.0
    eval_noise_scale=0.5
    reward_scale = 1.0
    action_range=20.
    state_dim = 182
    action_dim = 2
    hidden_dim = 128
    policy_target_update_interval = 3 # delayed update for the policy network and target networks
    DETERMINISTIC=True  # DDPG: deterministic policy gradient
    env_name = "./tac_real2"  # Name of the Unity environment binary to launch
    replay_buffer = ReplayBuffer(1e6)
    td3_trainer=TD3_Trainer(replay_buffer,state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim, policy_target_update_interval=3, action_range=20. )
    model_path = './model/td3_all'

    env = UnityEnv(env_name, worker_id=np.random.randint(0,10), use_visual=False, use_both=True)
    td3_trainer.load_model(model_path)
    batch_s = []
    cnt=0
    for eps in range(training_episodes):
        print(eps)
        s,info = env.reset()
        
        for step in range(episode_length):
            batch_s.append(s)
            s= state_process(s)
            a = td3_trainer.policy_net.get_action(s, deterministic = DETERMINISTIC, explore_noise_scale=0.0)
            a+=np.random.normal(0, 5, a.shape[0])  
            s_, r, d, _ = env.step(a)
            cnt+=1
            s=s_

            # # print(np.mean(np.abs(np.array(s[7:])-s0)))  # choose 0.03
            # if step >0 and np.mean(np.abs(np.array(s[7:])-s0))>0.03 and s[4]+s[5]+s[6]!=0:  # set a threshold to extract deformation frames
            #     batch_s.append(s)     # dim of s total 280 (select 182 as obs): 0 object index, 1-3 rotation value, 4-6 average contact point position, 7-279 pins positions
            #     cnt+=1

            
    print('total number of samples: ', cnt)
    pickle.dump(batch_s, data_file)
