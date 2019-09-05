"""
pure vector observation based learning: position of tactip and target
task: tactip following the cylinder to reach the ball target
use 382 pins
"""

import tensorflow as tf
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten
import numpy as np
import matplotlib.pyplot as plt
import gym, threading, queue
from gym_unity.envs import UnityEnv
import argparse
from PIL import Image
# from deform_visualize import plot_list_new
import pickle

data_file=open('data_all/random2_data_train.pickle', "rb")

parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=False)
parser.add_argument('--test', dest='test', action='store_true', default=False)

args = parser.parse_args()


class Classifier(object):
    def __init__(self, obs_dim, label_dim, z_dim, para_dim, ini_lr, num_obj=3):   
        self.hidden_dim=500  
        self.z_dim=z_dim
        self.para_dim = para_dim
        self.obs_dim=obs_dim
        self.label_dim = label_dim
        self.num_obj=num_obj        
        self.sess = tf.Session()

        self.label = tf.placeholder(tf.float32, [None, label_dim], 'label')  
        self.label_obj = tf.placeholder(tf.int8, [None, num_obj], 'label_obj')  
        self.obs = tf.placeholder(tf.float32, [None, obs_dim], 'obs')
        self.para = tf.placeholder(tf.float32, [None, para_dim], 'para')
        self.lr = tf.placeholder_with_default(ini_lr,  shape=(), name='lr')

        self.z = self.encoder()
        self.z_para = tf.concat([self.z, self.para], 1)
        self.recons = self.decoder()
        self.recon_loss = tf.losses.mean_squared_error(self.obs, self.recons)
        self.predict_obj, self.predict = self.predictor()

        self.predict_val_loss = tf.reduce_mean(tf.square(self.predict-self.label))
        self.predict_obj_loss = tf.reduce_mean(tf.square(self.predict_obj-tf.cast(self.label_obj, tf.float32)))
        self.loss = self.recon_loss + self.predict_val_loss + self.predict_obj_loss
        # self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)

        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        self.train_op = self.optimizer.minimize(self.loss)
            
        tf.summary.scalar('recon_loss', self.recon_loss)
        tf.summary.scalar('predict_val_loss', self.predict_val_loss)
        tf.summary.scalar('predict_obj_loss', self.predict_obj_loss)
        tf.summary.scalar('total_loss', self.loss)
        self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter( './log/train',
                                        self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def encoder(self, ):
        l1 = tf.layers.dense(self.obs, self.hidden_dim, tf.nn.relu)
        l2 = tf.layers.dense(l1, self.hidden_dim, tf.nn.relu)
        l3 = tf.layers.dense(l2, self.hidden_dim, tf.nn.relu)
        z = tf.layers.dense(l3, self.z_dim)
        return z

    def decoder(self, ):
        l1 = tf.layers.dense(self.z_para, self.hidden_dim, tf.nn.relu)
        l2 = tf.layers.dense(l1, self.hidden_dim, tf.nn.relu)
        l3 = tf.layers.dense(l2, self.hidden_dim, tf.nn.relu)
        recons = tf.layers.dense(l3, self.obs_dim)
        return recons

    def predictor(self, ):
        l1 = tf.layers.dense(self.z, self.hidden_dim, tf.nn.relu)
        l2 = tf.layers.dense(l1, self.hidden_dim, tf.nn.relu)
        l3 = tf.layers.dense(l2, self.hidden_dim, tf.nn.relu)
        predict = tf.layers.dense(l3, self.label_dim)  # predict position and rotation
        logits = tf.layers.dense(l2, self.num_obj, tf.nn.relu)
        predict_obj = tf.nn.softmax(logits)  # predict index of object

        return predict_obj, predict


    def train(self,  input_state, label_val, label_obj, para_list, lr, decay, idx):

        summary, recon_loss, predict_obj_loss, predict_val_loss, loss ,_=self.sess.run([self.merged, self.recon_loss, self.predict_obj_loss, self.predict_val_loss, self.loss, \
            self.train_op], { self.obs: input_state, self.label: label_val, self.label_obj: label_obj, self.para: para_list, self.lr: lr})
        self.train_writer.add_summary(summary, idx)

        # recon_loss, predict_obj_loss, predict_val_loss, loss ,_=self.sess.run([self.recon_loss, self.predict_obj_loss, self.predict_val_loss, self.loss, \
        #     self.train_op], { self.obs: input_state, self.label: label_val, self.label_obj: label_obj, self.para: para_list, self.lr: lr})

        return recon_loss, predict_obj_loss, predict_val_loss, loss

    def predict_one_value(self, s):
        s = s[np.newaxis, :]
        predict = self.sess.run(self.predict, {self.obs: s})
        predict_obj  =self.sess.run(self.predict_obj, {self.obs: s})
        return predict_obj, predict
        
    def predict_value(self, s):
        predict = self.sess.run(self.predict, {self.obs: s})
        predict_obj  =self.sess.run(self.predict_obj, {self.obs: s})
        return np.concatenate((predict_obj, predict), axis=1)

    def encode(self, s):
        z==self.sess.run(self.z, {self.obs: s})
        return z

    def save(self, path):
        saver = tf.train.Saver()
        saver.save(self.sess, path)

    def load(self, path):
        saver=tf.train.Saver()
        saver.restore(self.sess, path)


def plot(s):
    x=s[10::3]
    z=s[12::3]
    plot_list_new(x,z)

def state_process(s):
    ps=np.concatenate((s[10::3],s[12::3]))
    return ps


def to_one_hot(idx_list): # return one-hot vector list for object index predicting
    num_samples = len(idx_list)
    one_hot = np.zeros((num_samples, num_obj))
    one_hot[np.arange(num_samples), np.array(idx_list)] = 1

    return one_hot


def get_data(data_file):
    raw_data=pickle.load(data_file)
    para_list=[]
    label_val=[]
    label_obj=[]
    input_state=[]
    for i in range(len(raw_data)):
        s=raw_data[i]
        # object index: 1 dim
        label_obj.append(int(s[0]))

        rotat=s[1:4]/30.
        pos=np.concatenate(([s[4]], [s[6]]))
        para=s[7:10]/30.  # pushing, pulling, damping
        # value: 5 dim (3+2)
        label_val.append(np.concatenate((rotat, pos)))
        para_list.append(para)

        # input state: 182 dim
        data_i=state_process(s)
        ''' add noise '''
        data_i=data_i+np.random.normal(0, 1e-2, data_i.shape[0])
        input_state.append(data_i)

    label_obj=to_one_hot(label_obj)
    return label_val, label_obj, para_list, input_state


if __name__ == '__main__':
    model_path = './model/class_obj'
    training_episodes = 80000
    episode_length = 150
    obs_dim = 182  # total 280: 0 object index, 1-3 rotation value, 4-6 average contact point position, 7-279 pins positions
    val_dim = 5  # 3 rotation, 2 position
    para_dim = 3
    z_dim=20
    num_obj=3  # number of objects

    # lr=1e-3
    lr=2e-2
    decay=0 # decay signal of lr
    classifier = Classifier(obs_dim, val_dim, z_dim, para_dim, lr, num_obj)

    if args.train:
        label_val, label_obj, para_list, input_state=get_data(data_file)
        loss_list=[]
        classifier.load(model_path)

        for eps in range(training_episodes):
            if eps%40000==0 and eps>1:
                lr *=0.5
                decay=1
            else:
                decay=0
            recon_loss, predict_obj_loss, predict_val_loss, loss = classifier.train(input_state, label_val, label_obj, para_list, lr, decay, eps)
            if eps==0:
                loss_list.append(loss)
            else:
                loss_list.append(0.9*loss_list[-1]+0.1*loss)
            print('Eps: {}, Loss: {} {} {} {}'.format(eps, loss, recon_loss, predict_obj_loss,predict_val_loss))
            if eps % 100 ==0:
                plt.yscale('log')
                plt.plot(np.arange(len(loss_list)), loss_list)
                plt.savefig('classify_trainwithdataobj.png')
                classifier.save(model_path)
        
        np.savetxt('trainwithdata.txt', np.array(loss_list)[:, np.newaxis], fmt='%.4f', newline=', ')
        round_loss_list=list(np.around(np.array(loss_list),4))
        print(round_loss_list)


# test with training dataset, one by one
    # if args.test:
    #     raw_data=pickle.load(data_file)
    #     label=[]
    #     classifier.load(model_path)  

    #     for i in range(len(raw_data)):
    #         predict_obj, predict = classifier.predict_one_value(state_process(raw_data[i]))
    #         predict = np.concatenate((predict_obj[0],predict[0]))
    #         label_single= np.concatenate(([raw_data[i][4]], [raw_data[i][6]]))
    #         label =np.concatenate((raw_data[i][1:4]/30., label_single))
    #         label =np.concatenate((to_one_hot([int(raw_data[i][0])])[0], label))
    #         print('Pre: {}, Label: {}, Error: {}'.format(predict, label, np.average(np.square(predict-label))))
    #         # plot(raw_data[i], predict[0], label, idx=str(i))


# test with testing dataset, all at once
    if args.test:
        # test_data_file=open('data_all/fixed_data_train.pickle', "rb")
        test_data_file=open('data_all/random_data_test.pickle', "rb")
        # test_data_file=open('data_all/random2_data_test.pickle', "rb")

        label_val, label_obj, para_list, input_state=get_data(test_data_file)

        classifier.load(model_path)  

        predict = classifier.predict_value(input_state)
        label_list=np.concatenate((label_obj, label_val), axis=1)
        loss=np.mean(np.square(np.array(label_list)-np.array(predict)))
        loss_obj=np.mean(np.square(np.array(label_list)[:, :3]-np.array(predict)[:, :3]))
        loss_rotat=np.mean(np.square(np.array(label_list)[:, 3:6]-np.array(predict)[:, 3:6]))
        loss_pos=np.mean(np.square(np.array(label_list)[:, 6:]-np.array(predict)[:, 6:]))
        print('test loss: {:.4f}  {:.4f}  {:.4f}  {:.4f}'.format(loss_obj, loss_rotat, loss_pos, loss))