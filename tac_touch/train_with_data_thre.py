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

# data_file=open('data_thre/data_train.pickle', "rb")
# label_file=open('data_thre/label_train.pickle', "rb")

data_file=open('data_thre/random2_data_train.pickle', "rb")
label_file=open('data_thre/random2_label_train.pickle', "rb")

parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=False)
parser.add_argument('--test', dest='test', action='store_true', default=False)

args = parser.parse_args()


class Classifier(object):
    def __init__(self, obs_dim, label_dim, ini_lr, num_obj=3):   
        self.hidden_dim=500  
        self.num_obj=num_obj        
        self.sess = tf.Session()
        self.label = tf.placeholder(tf.float32, [None, label_dim], 'label')  
        self.label_obj = tf.placeholder(tf.int8, [None, num_obj], 'label_obj')  
        self.obs = tf.placeholder(tf.float32, [None, obs_dim], 'obs')
        self.lr = tf.placeholder_with_default(ini_lr,  shape=(), name='lr')
        self.training = tf.placeholder_with_default(False, shape=(), name='training')  # BN signal

        l1 = tf.layers.dense(self.obs, self.hidden_dim, tf.nn.relu)
        # l1 = tf.layers.batch_normalization(l1, training=self.training, momentum=0.9)
        l2 = tf.layers.dense(l1, self.hidden_dim, tf.nn.relu)
        # l2 = tf.layers.batch_normalization(l2, training=self.training, momentum=0.9)
        l3 = tf.layers.dense(l2, self.hidden_dim, tf.nn.relu)
        self.predict = tf.layers.dense(l3, label_dim)  # predict position and rotation
        logits = tf.layers.dense(l2, self.num_obj, tf.nn.relu)
        # logits = tf.layers.batch_normalization(logits, training=self.training, momentum=0.9)
        self.predict_obj = tf.nn.softmax(logits)  # predict index of object

        self.loss1 = tf.reduce_mean(tf.square(self.predict-self.label))  # predict position, angles
        self.loss2 = tf.reduce_mean(tf.square(self.predict_obj-tf.cast(self.label_obj, tf.float32)))  # predict 
        self.loss = self.loss1 + self.loss2
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        self.train_op = self.optimizer.minimize(self.loss)
        self.sess.run(tf.global_variables_initializer())

    def train(self,  batch_s, batch_label, batch_label_obj, lr, decay):
        loss,_=self.sess.run([self.loss, self.train_op], {self.training: True, self.obs: batch_s, self.label: batch_label, self.label_obj: batch_label_obj, self.lr: lr})
        return loss

    def predict_one_value(self, s):
        s = s[np.newaxis, :]
        predict = self.sess.run(self.predict, {self.obs: s})
        predict_obj  =self.sess.run(self.predict_obj, {self.obs: s})
        return predict_obj, predict

    def predict_value(self, s):
        predict = self.sess.run(self.predict, {self.obs: s})
        predict_obj  =self.sess.run(self.predict_obj, {self.obs: s})
        return np.concatenate((predict_obj, predict), axis=1)

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


def state_process(s, s0):
    # for threshold representation
    dis_threshold=0.02  # lower bound threshold, change data_collect_thre.py at the same time!
    x0=s0[7::3]
    z0=s0[9::3]

    x=s[7::3]
    z=s[9::3]
    dis=np.abs((x-x0))+np.abs((z-z0))
    dis_threshold = max(1.2*np.average(dis), dis_threshold)
    dis_idx=np.argwhere(dis>dis_threshold).reshape(-1)
    dis[dis<=dis_threshold]=0
    dis[dis>dis_threshold]=1
    processed_state=dis
    return processed_state


def to_one_hot(idx_list): # return one-hot vector list for object index predicting
    num_samples = len(idx_list)
    one_hot = np.zeros((num_samples, num_obj))
    one_hot[np.arange(num_samples), np.array(idx_list)] = 1

    return one_hot


if __name__ == '__main__':
    model_path = './model/class_obj_thre'
    training_episodes = 80000
    episode_length = 150
    obs_dim = 91  # total 280: 0 object index, 1-3 rotation value, 4-6 average contact point position, 7-279 pins positions
    state_dim = 5
    lr=2e-2
    decay=0 # decay signal of lr
    num_obj=3  # number of objects
    classifier = Classifier(obs_dim, state_dim, lr, num_obj)

    if args.train:
        data=pickle.load(data_file)
        for i in range(len(data)):
            data[i]=data[i]+np.random.normal(0, 1e-2, data[i].shape[0])  # add noise

        label_data=pickle.load(label_file)
        label=[]
        label_obj=[]
        for i in range(len(label_data)):
            label.append(label_data[i][1:6])  # normalize the rotation range by 30, to get [-1,1]
            label_obj.append(int(label_data[i][0]))
        label_obj=to_one_hot(label_obj)
        loss_list=[]
        # classifier.load(model_path)

        for eps in range(training_episodes):
            if eps%40000==0 and eps>1:
                lr *=0.5
                decay=1
            else:
                decay=0
            loss = classifier.train(data, label, label_obj, lr, decay)
            if eps==0:
                loss_list.append(loss)
            else:
                loss_list.append(0.9*loss_list[-1]+0.1*loss)
            print('Eps: {}, Loss: {}'.format(eps, loss))
            if eps % 100 ==0:
                plt.yscale('log')
                plt.plot(np.arange(len(loss_list)), loss_list)
                plt.savefig('classify_trainwithdataobj_thre.png')
                classifier.save(model_path)
        np.savetxt('trainwithdata_thre.txt', np.array(loss_list)[:, np.newaxis], fmt='%.4f', newline=', ')
        round_loss_list=list(np.around(np.array(loss_list),4))
        print(round_loss_list)

    # if args.test:
    #     env_name = "./tac_touch_fixed"  # Name of the Unity environment binary to launch
    #     env = UnityEnv(env_name, worker_id=np.random.randint(0,10), use_visual=False, use_both=True)

    #     env.reset()
    #     classifier.load(model_path)  # if re-train
    #     test_steps = 150
    #     test_episode = 10
        
    #     total_error_list=[]
    #     for _ in range(test_episode):
    #         s,info = env.reset()
    #         s0=np.array(s)
    #         for step in range(episode_length):
    #             if step >0 and np.mean(np.abs(np.array(s[7:])-s0[7:]))>0.6 and s[4]+s[5]+s[6]!=0:  # should be same as in data_collect!
    #                 predict_obj, predict = classifier.predict_value(state_process(s,s0))
    #                 label = np.concatenate((s[1:4]/30., s[4:7]))
    #                 # label = np.concatenate(([s[0]], label))
    #                 # predict = np.concatenate(([np.argmax(predict_obj)],predict[0]))
    #                 # print('Pre: {}, Label: {}'.format(predict, label ))
    #                 label = np.concatenate((to_one_hot([int(s[0])])[0], label))
    #                 predict = np.concatenate((predict_obj[0],predict[0]))
    #                 print('Pre: {}, Label: {}'.format(predict, label ))
    #                 error = np.average(np.abs(predict-label))
    #                 total_error_list.append(error)
    #             s_, r, done, info= env.step([0])
    #             s=s_

    #     print(np.mean(total_error_list), np.std(total_error_list))




# test with testing dataset, all at once
    if args.test:
        # data_file=open('data_thre/data_train.pickle', "rb")
        # label_file=open('data_thre/label_train.pickle', "rb")
        # data_file=open('data_thre/data_test.pickle', "rb")
        # label_file=open('data_thre/label_test.pickle', "rb")
        data_file=open('data_thre/random_data_test.pickle', "rb")
        label_file=open('data_thre/random_label_test.pickle', "rb")
        # data_file=open('data_thre/random2_data_test.pickle', "rb")
        # label_file=open('data_thre/random2_label_test.pickle', "rb")

        data=pickle.load(data_file)
        label_data=pickle.load(label_file)
        label=[]
        label_obj=[]
        for i in range(len(label_data)):
            label.append(label_data[i][1:6])  # normalize the rotation range by 30, to get [-1,1]
            label_obj.append(int(label_data[i][0]))
        label_obj=to_one_hot(label_obj)
        label_list=np.concatenate((label_obj, label), axis=1)
    
        classifier.load(model_path)  

        predict = classifier.predict_value(data)
        loss=np.mean(np.square(np.array(label_list)-np.array(predict)))
        print(predict.shape, predict[0])
        loss_obj=np.mean(np.square(np.array(label_list)[:, :3]-np.array(predict)[:, :3]))
        loss_rotat=np.mean(np.square(np.array(label_list)[:, 3:6]-np.array(predict)[:, 3:6]))
        loss_pos=np.mean(np.square(np.array(label_list)[:, 6:]-np.array(predict)[:, 6:]))
        print('test loss: {:.4f}  {:.4f}  {:.4f}  {:.4f}'.format(loss_obj, loss_rotat, loss_pos, loss))