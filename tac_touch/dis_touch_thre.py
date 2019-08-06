"""
pure vector observation based learning: position of tactip and target
task: tactip following the cylinder to reach the ball target
use 382 pins
"""

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
import numpy as np
import matplotlib.pyplot as plt
import gym, threading, queue
from gym_unity.envs import UnityEnv
import argparse
from PIL import Image
from deform_visualize import plot_list_new

parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=False)
parser.add_argument('--test', dest='test', action='store_true', default=False)

args = parser.parse_args()


class Classifier(object):
    def __init__(self, obs_dim, label_dim):   
        self.hidden_dim=100          
        self.sess = tf.Session()
        self.label = tf.placeholder(tf.float32, [None, label_dim], 'label')  
        self.obs = tf.placeholder(tf.float32, [None, obs_dim], 'label')
        self.lr = 5e-5 # 2e-4

        l1 = tf.layers.dense(self.obs, self.hidden_dim, tf.nn.relu)
        l2 = tf.layers.dense(l1, self.hidden_dim, tf.nn.relu)
        l3 = tf.layers.dense(l2, self.hidden_dim, tf.nn.relu)
        self.predict = tf.layers.dense(l3, label_dim)

        self.loss = tf.reduce_mean(tf.square(self.predict-self.label))
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        self.sess.run(tf.global_variables_initializer())


    def train(self,  batch_s, batch_label):
        loss,_=self.sess.run([self.loss, self.train_op], {self.obs: batch_s, self.label: batch_label})
        return loss

    def predict_label(self, s):
        s = s[np.newaxis, :]
        predict = self.sess.run(self.predict, {self.obs: s})
        return predict

    def save(self, path):
        saver = tf.train.Saver()
        saver.save(self.sess, path)

    def load(self, path):
        saver=tf.train.Saver()
        saver.restore(self.sess, path)

def plot(s):
    x=s[1::3]
    z=s[3::3]
    plot_list_new(x,z)

def state_process(s, s0):
    dis_threshold=0.05

    x0=s0[1::3]
    z0=s0[3::3]


    x=s[1::3]
    z=s[3::3]
    dis=np.abs(x-x0)+np.abs(z-z0)
    dis_threshold = max(1.2*np.average(dis), dis_threshold)
    dis_idx=np.argwhere(dis>dis_threshold).reshape(-1)
    dis[dis<=dis_threshold]=0
    dis[dis>dis_threshold]=1
    # print(dis_idx)
    # plt.figure(figsize=(5,4))  # this line cause memory keeping increasing if not close the figure

    # plot_list_new(x-object_x, z-object_z, dis_idx)  
    processed_state=dis
    return processed_state


if __name__ == '__main__':
    model_path = './model/class'
    training_episodes = 100000
    episode_length = 150
    obs_dim = 91
    state_dim = 1
    classifier = Classifier(obs_dim, state_dim)
    env_name = "./tac_touch_fixed"  # Name of the Unity environment binary to launch


    env = UnityEnv(env_name, worker_id=np.random.randint(0,10), use_visual=False, use_both=True)

    if args.train:
        loss_list=[]
        # classifier.load(model_path)

        for eps in range(training_episodes):
            batch_s = []
            batch_label = []
            s,info = env.reset()
            s0=s
            for step in range(episode_length):
                # plot(s)
                if step >0 and np.mean(np.abs(np.array(s[1:])-np.array(s0[1:])))>0.15:  # set a threshold to extract deformation frames
                    ps=state_process(s,s0)
                    batch_s.append(ps)
                    batch_label.append([s[0]])  # predict number of collision points

                s_, r, done, info= env.step([0])
                s=s_

                # if done:  # done signal not work in the env
                #     break
            if len(batch_s)>0:
                loss = classifier.train(batch_s, batch_label)
                if eps==0:
                    loss_list.append(loss)
                else:
                    loss_list.append(0.9*loss_list[-1]+0.1*loss)
                print('Eps: {}, Loss: {}'.format(eps, loss))
            if eps % 10 ==0:
                plt.plot(np.arange(len(loss_list)), loss_list)
                plt.savefig('classify.png')
                classifier.save(model_path)

    if args.test:
        env.reset()
        classifier.load(model_path)
        test_steps = 150
        test_episode = 10
        

        for _ in range(test_episode):
            s,info = env.reset()
            s0=s
            for step in range(episode_length):
                if step >0 and np.mean(np.abs(np.array(s[1:])-np.array(s0[1:])))>0.15:
                    ps=state_process(s,s0)
                    predict = classifier.predict_label(ps)
                    print('Pre: {}, Label: {}'.format(predict, s[0]))

                s_, r, done, info= env.step([0])
                s=s_

        # print('Eps: {}, Loss: {}'.format(eps, loss))

