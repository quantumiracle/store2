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

parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=False)
parser.add_argument('--test', dest='test', action='store_true', default=False)

args = parser.parse_args()


class Classifier(object):
    def __init__(self, obs_dim, label_dim):   
        self.hidden_dim=100   
        self.lr = 2e-4        
        self.sess = tf.Session()
        self.label = tf.placeholder(tf.float32, [None, label_dim], 'label')  
        self.obs = tf.placeholder(tf.float32, [None, obs_dim], 'label')

        l1 = tf.layers.dense(self.obs, self.hidden_dim, tf.nn.relu)
        l2 = tf.layers.dense(l1, self.hidden_dim, tf.nn.relu)
        l3 = tf.layers.dense(l2, self.hidden_dim, tf.nn.relu)
        self.predict = tf.layers.dense(l3, self.hidden_dim, tf.nn.tanh)

        self.loss = tf.reduce_mean(tf.square(self.predict-self.label))
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        self.sess.run(tf.global_variables_initializer())


    def train(self,  batch_s, batch_label):
        loss,_=self.sess.run([self.loss, self.train_op], {self.obs: batch_s, self.label: batch_label})
        return loss

    def predict(self, s):
        s = s[np.newaxis, :]
        predict = self.sess.run(self.obs, {self.obs: s})
        return predict

    def save(self, path):
        saver = tf.train.Saver()
        saver.save(self.sess, path)

    def load(self, path):
        saver=tf.train.Saver()
        saver.restore(self.sess, path)


if __name__ == '__main__':
    model_path = './model/class'
    training_episodes = 1000
    episode_length = 150
    obs_dim = 274
    state_dim = 1
    classifier = Classifier(obs_dim, state_dim)
    env_name = "./tac_touch"  # Name of the Unity environment binary to launch


    env = UnityEnv(env_name, worker_id=np.random.randint(0,10), use_visual=False, use_both=True)

    if args.train:
        loss_list=[]
        for eps in range(training_episodes):
            batch_s = []
            batch_label = []
            s,info = env.reset()
            for step in range(episode_length):
                batch_s.append(s[1:])
                if s[0]>0: # predict collision or not 
                    batch_label.append([1])
                else: 
                    batch_label.append([-1])
                # batch_label.append([s[0]])  # predict number of collision points

                s_, r, done, info= env.step([0,0])
                s=s_

            loss = classifier.train(batch_s, batch_label)
            loss_list.append(loss)
            print('Eps: {}, Loss: {}'.format(eps, loss))
            if eps % 50 ==0:
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
            for step in range(episode_length):
                batch_s.append(s[1:])
                if s[0]>0: # predict collision or not 
                    batch_label.append([1])
                else: 
                    batch_label.append([0])
                # batch_label.append([s[0]])  # predict number of collision points

                predict = classifier.predict(s)
                print('Pre: {}, Label: {}'.format(predict, s[0]))

                s_, r, done, info= env.step([0,0,0,0,0,0])
                s=s_

        # print('Eps: {}, Loss: {}'.format(eps, loss))

