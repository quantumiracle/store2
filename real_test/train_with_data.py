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
from deform_visualize import plot_list_new, plot_list_new_sim2
import pickle


parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=False)
parser.add_argument('--test', dest='test', action='store_true', default=False)
parser.add_argument('--rotat_test', dest='rotat_test', action='store_true', default=False)

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
        # l31 = tf.layers.dense(l3, self.hidden_dim, tf.nn.relu)
        self.predict = tf.layers.dense(l3, label_dim)  # predict position and rotation
        # self.predict = tf.layers.batch_normalization(self.predict, training=training, momentum=0.9)
        # l21 = tf.layers.dense(l2, self.hidden_dim, tf.nn.relu)
        logits = tf.layers.dense(l2, self.num_obj, tf.nn.relu)
        # logits = tf.layers.batch_normalization(logits, training=self.training, momentum=0.9)
        self.predict_obj = tf.nn.softmax(logits)  # predict index of object
        # self.loss1 = tf.reduce_mean(tf.square(self.predict[:, :3]-self.label[:, :3]))  # rotation
        self.loss1 = tf.reduce_mean(tf.square(self.predict[:, 3:]-self.label[:, 3:]))  # pos

        # self.loss2 = tf.reduce_mean(tf.square(self.predict_obj-tf.cast(self.label_obj, tf.float32)))
        # self.loss = self.loss1 + self.loss2
        self.loss = self.loss1
        # self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        self.train_op = self.optimizer.minimize(self.loss)
        self.sess.run(tf.global_variables_initializer())

    def train(self,  batch_s, batch_label, batch_label_obj, lr, decay):
        # self.optimizer.learning_rate = lr
        # if decay:
        #     self.train_op = self.optimizer.minimize(self.loss)
        loss,_=self.sess.run([self.loss, self.train_op], {self.training: True, self.obs: batch_s, self.label: batch_label, self.label_obj: batch_label_obj, self.lr: lr})
        # if decay: 
        #     print(self.optimizer._lr)
        return loss

    def predict_one_value(self, s):
        s = s[np.newaxis, :]
        predict = self.sess.run(self.predict, {self.obs: s})
        predict_obj  =self.sess.run(self.predict_obj, {self.obs: s})
        return np.concatenate((predict_obj, predict), axis=1)
        
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

def state_process(s):
    ps=np.concatenate((s[7::3],s[9::3]))   # ((x,x,x..),(y,y,y,...))
    return ps


def to_one_hot(idx_list): # return one-hot vector list for object index predicting
    num_samples = len(idx_list)
    # print(idx_list.shape)
    # print(num_samples, self.num_obj)
    one_hot = np.zeros((num_samples, num_obj))
    one_hot[np.arange(num_samples), np.array(idx_list)] = 1

    return one_hot


def Predict(input, model_path = './model/class_obj'): 
    obs_dim = 182  # total 280: 0 object index, 1-3 rotation value, 4-6 average contact point position, 7-279 pins positions
    state_dim = 5  # 3 rotation, 2 position
    lr=2e-2
    num_obj=3  # number of objects
    classifier = Classifier(obs_dim, state_dim, lr, num_obj)
    classifier.load(model_path) 
    predict = classifier.predict_one_value(input)
    return predict

if __name__ == '__main__':
    model_path = './model/class_obj'
    training_episodes = 80000
    episode_length = 150
    obs_dim = 182  # total 280: 0 object index, 1-3 rotation value, 4-6 average contact point position, 7-279 pins positions
    state_dim = 5  # 3 rotation, 2 position
    # lr=1e-3
    lr=2e-2
    decay=0 # decay signal of lr
    num_obj=3  # number of objects
    classifier = Classifier(obs_dim, state_dim, lr, num_obj)

    if args.train:
        # data_file=open('data_all/fixed_data.pickle', "rb")
        data_file=open('data_all/random2_data_train.pickle', "rb")

        raw_data=pickle.load(data_file)
        data=[]
        label=[]
        label_obj=[]
        
        for i in range(len(raw_data)):
            s=raw_data[i]
            data_i=state_process(s)
            ''' add noise '''
            data_i=data_i+np.random.normal(0, 1e-2, data_i.shape[0])  
            data.append(data_i)
            label_pos= np.concatenate(([s[4]], [s[6]]))
            label.append(np.concatenate((s[1:4]/30., label_pos)))  # normalize the rotation range by 30, to get [-1,1]
            label_obj.append(int(s[0]))

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
                plt.savefig('classify_trainwithdataobj.png')
                classifier.save(model_path)
        
        np.savetxt('trainwithdata.txt', np.array(loss_list)[:, np.newaxis], fmt='%.4f', newline=', ')
        round_loss_list=list(np.around(np.array(loss_list),4))
        print(round_loss_list)




# test with testing dataset, all at once
    if args.test:
        # test_data_file=open('data_all/data_train.pickle', "rb")
        # test_data_file=open('data_all/fixed_data.pickle', "rb")
        test_data_file=open('data_all/random2_data_train.pickle', "rb")
        # test_data_file=open('data_all/random2_data_test.pickle', "rb")

        raw_data=pickle.load(test_data_file)
        data=[]
        label_list=[]
        classifier.load(model_path)  
        for i in range(len(raw_data)):
            s=raw_data[i]
            # print('x:', s[7::3])
            # print('y: ', s[9::3])
            data.append(state_process(s))
            label_single= np.concatenate(([raw_data[i][4]], [raw_data[i][6]]))
            label=np.concatenate((raw_data[i][1:4]/30., label_single))
            label_list.append( np.concatenate((to_one_hot([int(raw_data[i][0])])[0], label)) )
        
        predict = classifier.predict_value(data)
        loss=np.mean(np.square(np.array(label_list)-np.array(predict)))
        loss_obj=np.mean(np.square(np.array(label_list)[:, :3]-np.array(predict)[:, :3]))
        loss_rotat=np.mean(np.square(np.array(label_list)[:, 3:6]-np.array(predict)[:, 3:6]))
        loss_pos=np.mean(np.square(np.array(label_list)[:, 6:]-np.array(predict)[:, 6:]))
        print('test loss: {:.4f}  {:.4f}  {:.4f}  {:.4f}'.format(loss_obj, loss_rotat, loss_pos, loss))

        norm=0.5674
        for i in range(80):
            # data_i=data[i]+np.random.normal(0, 1e-2, data[i].shape[0])
            data_i=data[i]
            xy=np.transpose(data_i.reshape(2, -1))/norm
            pos_label=[raw_data[i][4]], [raw_data[i][6]]
            plot_list_new_sim2(xy,i,predict[i][6:]/norm, np.array(pos_label)/norm)
            print(i)

    if args.rotat_test:
        # test_data_file=open('data_all/data_train.pickle', "rb")
        # test_data_file=open('data_all/data_test.pickle', "rb")
        test_data_file=open('data_all/fixed_data.pickle', "rb")
        # test_data_file=open('data_all/random2_data_test.pickle', "rb")

        raw_data=pickle.load(test_data_file)
        data=[]
        label_list=[]
        classifier.load(model_path)  
        for i in range(len(raw_data)):
            s=raw_data[i]
            data.append(state_process(s))
            label_single= np.concatenate(([raw_data[i][4]], [raw_data[i][6]]))
            label=np.concatenate((raw_data[i][1:4]/30., label_single))
            label_list.append( np.concatenate((to_one_hot([int(raw_data[i][0])])[0], label)) )
        
        predict = classifier.predict_value(data)
        loss=np.mean(np.square(np.array(label_list)-np.array(predict)))
        loss_obj=np.mean(np.square(np.array(label_list)[:, :3]-np.array(predict)[:, :3]))
        loss_rotat=np.mean(np.square(np.array(label_list)[:, 3:6]-np.array(predict)[:, 3:6]))
        loss_pos=np.mean(np.square(np.array(label_list)[:, 6:]-np.array(predict)[:, 6:]))
        print('test loss: {:.4f}  {:.4f}  {:.4f}  {:.4f}'.format(loss_obj, loss_rotat, loss_pos, loss))

        norm2sim=30.
        for i in range(40):
            print(predict[i][3:6], label_list[i][3:6])
            # plot_list_new_sim(xy,i,predict[i][6:]/norm2sim)
            # print(i)