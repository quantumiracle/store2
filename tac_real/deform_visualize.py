import gym
import numpy as np
import matplotlib.pyplot as plt


def plot_two(min_idx, max_idx):

    x=np.array([5,7,9,11,13,15,
    4,6,8,10,12,14,16,
    3,5,7,9,11,13,15,17,
    2, 4,6,8,10,12,14,16,18,
    1,3,5,7,9,11,13,15,17,19,
    0,2,4,6,8,10,12,14,16,18,20,
    1,3,5,7,9,11,13,15,17,19,
    2, 4,6,8,10,12,14,16,18,
    3,5,7,9,11,13,15,17,
    4,6,8,10,12,14,16,
    5,7,9,11,13,15
    ])

    y=np.array([
        10,10,10,10,10,10,
        9,9,9,9,9,9,9,
        8,8,8,8,8,8,8,8,
        7,7,7,7,7,7,7,7,7,
        6,6,6,6,6,6,6,6,6,6,
        5,5,5,5,5,5,5,5,5,5,5,
        4,4,4,4,4,4,4,4,4,4,
        3,3,3,3,3,3,3,3,3,
        2,2,2,2,2,2,2,2,
        1,1,1,1,1,1,1,
        0,0,0,0,0,0
    ])

    plt.figure(figsize=(5,4))
    plt.scatter(x,y, c='b')
    plt.scatter(x[min_idx], y[min_idx], c='g')
    plt.scatter(x[max_idx], y[max_idx], c='r')
    plt.savefig('./deform.png')
    plt.show()
    plt.pause(0.1)

def plot_one(xx, yy):

    x=np.array([5,7,9,11,13,15,
    4,6,8,10,12,14,16,
    3,5,7,9,11,13,15,17,
    2, 4,6,8,10,12,14,16,18,
    1,3,5,7,9,11,13,15,17,19,
    0,2,4,6,8,10,12,14,16,18,20,
    1,3,5,7,9,11,13,15,17,19,
    2, 4,6,8,10,12,14,16,18,
    3,5,7,9,11,13,15,17,
    4,6,8,10,12,14,16,
    5,7,9,11,13,15
    ])

    y=np.array([
        10,10,10,10,10,10,
        9,9,9,9,9,9,9,
        8,8,8,8,8,8,8,8,
        7,7,7,7,7,7,7,7,7,
        6,6,6,6,6,6,6,6,6,6,
        5,5,5,5,5,5,5,5,5,5,5,
        4,4,4,4,4,4,4,4,4,4,
        3,3,3,3,3,3,3,3,3,
        2,2,2,2,2,2,2,2,
        1,1,1,1,1,1,1,
        0,0,0,0,0,0
    ])

    plt.figure(figsize=(5,4))
    plt.scatter(x,y, c='b')
    plt.scatter(xx, yy, c='g')
    plt.savefig('./deform.png')
    plt.show()
    plt.pause(0.1)

def plot_list(list):
    plt.ion()
    x=np.array([5,7,9,11,13,15,
    4,6,8,10,12,14,16,
    3,5,7,9,11,13,15,17,
    2, 4,6,8,10,12,14,16,18,
    1,3,5,7,9,11,13,15,17,19,
    0,2,4,6,8,10,12,14,16,18,20,
    1,3,5,7,9,11,13,15,17,19,
    2, 4,6,8,10,12,14,16,18,
    3,5,7,9,11,13,15,17,
    4,6,8,10,12,14,16,
    5,7,9,11,13,15
    ])

    y=np.array([
        10,10,10,10,10,10,
        9,9,9,9,9,9,9,
        8,8,8,8,8,8,8,8,
        7,7,7,7,7,7,7,7,7,
        6,6,6,6,6,6,6,6,6,6,
        5,5,5,5,5,5,5,5,5,5,5,
        4,4,4,4,4,4,4,4,4,4,
        3,3,3,3,3,3,3,3,3,
        2,2,2,2,2,2,2,2,
        1,1,1,1,1,1,1,
        0,0,0,0,0,0
    ])

    plt.figure(figsize=(5,4))
    plt.scatter(x,y, c='b')
    if len(list)>0:
        for i in list:
            plt.scatter(x[i], y[i], c='g')
    plt.savefig('./deform.png')
    plt.show()
    plt.pause(0.1)
    plt.close()



def plot_one_new(x,y, one_point):
    plt.ion()

    plt.clf()
    plt.scatter(-y,x, c='b')   # match with the scene in Unity

    plt.scatter(-one_point[1], one_point[0], c='g')
    plt.savefig('./deform.png')
    plt.show()
    plt.pause(0.1)
    plt.close()



def plot_list_new(list=[], idx=0, point=[]):
    plt.ion()

    plt.clf()
    if len(list)>0:
        for i in list:
            plt.scatter(i[0], i[1], c='g')

    if len(point)>0:
        plt.scatter(point[0], point[1], c='b')
    # plt.savefig('./img_real/deform' + str(idx)+'.png')
    plt.savefig('./test/deform' + str(idx)+'.png')
    # plt.show()
    # plt.pause(0.1)
    plt.close()

# def plot_list_new_sim(list=[], idx=0, point=[]):
#     plt.ion()

#     plt.clf()
#     if len(list)>0:
#         for i in list:
#             plt.scatter(-i[1], i[0], c='g')

#     if len(point)>0:
#         plt.scatter(-point[1], point[0], c='b')
#     # plt.savefig('./img_sim/deform' + str(idx)+'.png')
#     plt.savefig('./test_sim/deform' + str(idx)+'.png')
#     # plt.show()
#     # plt.pause(0.1)
#     plt.close()

def plot_list_new_sim(list=[], idx=0, point=[]):
    plt.ion()

    plt.clf()
    if len(list)>0:
        for i in list:
            plt.scatter(i[0], i[1], c='g')

    if len(point)>0:
        plt.scatter(point[0], point[1], c='b')
    # plt.savefig('./img_sim/deform' + str(idx)+'.png')
    plt.savefig('./test_sim/deform' + str(idx)+'.png')
    # plt.show()
    # plt.pause(0.1)
    plt.close()

def plot_list_new_sim2(list=[], idx=0, point=[], point_=[]):
    plt.ion()

    plt.clf()
    if len(list)>0:
        for i in list:
            plt.scatter(i[0], i[1], c='g')

    if len(point)>0:
        plt.scatter(point[0], point[1], c='b')

    if len(point)>0:
        plt.scatter(point_[0], point_[1], c='r')
    # plt.savefig('./img_sim/deform' + str(idx)+'.png')
    plt.savefig('./test_sim/deform' + str(idx)+'.png')
    # plt.show()
    # plt.pause(0.1)
    plt.close()

if __name__ == '__main__':
    # plot_two(2,23)
    # plot_one(2,8)
    plot_list([1,2])