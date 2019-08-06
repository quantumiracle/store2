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

data_file=open('data_fixed/data.pickle', "rb")
label_file=open('data_fixed/label.pickle', "rb")

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
        self.lr = 8e-4 # 2e-4

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
    x=s[7::3]
    z=s[9::3]
    plot_list_new(x,z)

def state_process(s):
    ps=np.concatenate((s[7::3],s[9::3]))
    return ps

if __name__ == '__main__':
    model_path = './model/class'
    training_episodes = 10000
    episode_length = 150
    obs_dim = 182  # total 280: 0 object index, 1-3 rotation value, 4-6 average contact point position, 7-279 pins positions
    state_dim = 6
    classifier = Classifier(obs_dim, state_dim)

    if args.train:
        data=pickle.load(data_file)
        label=pickle.load(label_file)
        loss_list=[]
        classifier.load(model_path)

        for eps in range(training_episodes):

            loss = classifier.train(data, label)
            if eps==0:
                loss_list.append(loss)
            else:
                loss_list.append(0.9*loss_list[-1]+0.1*loss)
            print('Eps: {}, Loss: {}'.format(eps, loss))
            if eps % 10 ==0:
                plt.plot(np.arange(len(loss_list)), loss_list)
                plt.savefig('classify_trainwithdata.png')
                classifier.save(model_path)

    if args.test:
        env_name = "./tac_touch_fixed"  # Name of the Unity environment binary to launch
        env = UnityEnv(env_name, worker_id=np.random.randint(0,10), use_visual=False, use_both=True)

        env.reset()
        # classifier.load(model_path)  # if re-train
        test_steps = 150
        test_episode = 10
        
        total_error_list=[]
        for _ in range(test_episode):
            s,info = env.reset()
            s0=np.array(s[7:])
            for step in range(episode_length):
                if step >0 and np.mean(np.abs(np.array(s[7:])-s0))>0.6 and s[4]+s[5]+s[6]!=0:  # should be same as in data_collect!
                    predict = classifier.predict_label(state_process(s))
                    label = np.concatenate((s[1:4]/30., s[4:7]))
                    print('Pre: {}, Label: {}'.format(predict, label ))
                    error = np.abs(np.average(predict-label))
                    total_error_list.append(error)
                s_, r, done, info= env.step([0])
                s=s_

        print(np.mean(total_error_list), np.std(total_error_list))