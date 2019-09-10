"""
Real Image Processing
drop the most outside circle of pins
https://www.learnopencv.com/blob-detection-using-opencv-python-c/

pins registration with relative angles to the center, circle by circle
"""
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
import time
import math
from deform_visualize import plot_list_new

def ImageProcessing(img):
    ''' read image and transfer to color image '''
    # cv2.imshow('origin',img)
    cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

    ''' get threshold, for contour detection in next step '''
    ret,thresh_img = cv2.threshold(img,80,255,cv2.THRESH_TOZERO)  # lower than low-threshold to be black
    # cv2.imshow('Threshold', thresh_img)

    ''' get canny edge image, for the large circle detection with HoughCricles '''
    edge_detected_image = cv2.Canny(thresh_img, 60, 200)
    # cv2.imshow('Canny', edge_detected_image)

    ''' find image contours and plot '''
    image, contours, hierarchy = cv2.findContours(thresh_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)


    contour_centers=[]  # detected contour centers list
    min_countour_size = 4  # minimal number of pixels for one contour, to exclude the very small wrong points
    max_countour_size = 20  # maximal number of pixels for one contour, to exclude the very large wrong points
    i=0
    pin_contours=[]
    # print(len(contours))
    while i < (len(contours)):
        if len(contours[i])>=min_countour_size and len(contours[i])<max_countour_size:
            contour_centers.append(np.average(contours[i], axis = 0).reshape(2))  # reshape for reducing dimension
            pin_contours.append(contours[i])
        # if len(contour_centers)==127:
        #     break
        i+=1
    cimg = cv2.drawContours(cimg, pin_contours, -1, (0,0,255), 1)

    if len(contour_centers)<127:
        image = cv2.drawContours(cimg, contours, -1, (0,0,255), 1)
        cv2.imshow('Contour', image)
        cv2.waitKey()

    try:
        assert len(contour_centers) >= 127
    except AssertionError as error:
        print('number of contours after image processing: ', len(contour_centers))
    return cimg, edge_detected_image, contour_centers


def ContourCenterCheck(contour_centers, cimg, NUM_PINS=127):
    ''' check the contour center detection and draw centers '''
    last_point=[[0,0]] 
    min_mse = 42 # minimum of mse for two points; this value is carefully chosen here
    max_mse = 250  # no collision around 130, with collision around 200
    checked_contour_centers = []

    ''' fast but inaccurate '''
    # for i in range(len(contour_centers)):
    #     mse = 1000  # large enough value

    #     ''' if two arbitrary points are too close, remove it'''
    #     for j in range(i+1, len(contour_centers)):
    #         distance = (np.square(contour_centers[i]-contour_centers[j])).mean()
    #         mse = distance if distance<mse else mse  # get minimal distance
        
    #     # ''' if two consecutive points are too close, remove it'''
    #     # mse = (np.square(round_point - last_point)).mean() 
    #     # print(mse)
    #     if (mse>min_mse and mse<max_mse) or (len(checked_contour_centers) == len(contour_centers)-1):  # minimal mse within valid range, and preserve the last one
    #         checked_contour_centers.append(contour_centers[i])
    #     # last_point = round_point

    '''slow but accurate '''
    i=0
    while i<len(contour_centers):
        mse = 1000  # large enough value
        if len(contour_centers) == 127:
            break
        ''' if two arbitrary points are too close, remove it'''
        for j in range(len(contour_centers)):
            distance = (np.square(contour_centers[i]-contour_centers[j])).mean()
            mse = distance if distance<mse and distance>0 else mse  # get minimal distance

        if mse<min_mse or mse>max_mse:
            contour_centers.pop(i)
            i-=1


        i+=1
        # print(i, mse)

    checked_contour_centers = contour_centers

    try:
        assert len(checked_contour_centers) == 127
    except AssertionError as error:
        print('number of contours after contour center check: ', len(checked_contour_centers))
    return cimg, checked_contour_centers


def CenterRegister(contour_centers, cimg):
    '''
    deprecated! register all pins, according to their x, y coordinates
    '''
    ordered= [[0,0]] 
    num_list = [ 7,8,9,10,11,12,13,12,11,10,9,8,7]  
    contour_centers=np.array(contour_centers)
    for i in range(len(num_list)):
        indx=np.argsort(-contour_centers[:,-1])[:num_list[i]]  # sort on y-axis, select top-jth largest, return index
        # print('ind: ', indx)
        y_contour = contour_centers[indx.tolist()]
        x_contour = y_contour[y_contour[:, 0].argsort()]  # sort on x-axis
        # print('x: ', x_contour)
        ordered=np.concatenate((ordered, x_contour))
        cimg = PlotCenters(x_contour, cimg)
        cv2.imshow('Register', cimg)
        # cv2.waitKey(0)
        contour_centers=np.delete(contour_centers, indx.tolist(), axis=0)
        # print('len: ', len(contour_centers))
    # print(ordered[1:])  # remove the first [0,0]

    return ordered[1:]



def PointCheck(contour_centers, last_contour_centers, max_dis, smooth_factor=0.1):
    ''' 
    restrict the displacement for each pin on consecutive frame images,
    if larger than max_dis, keep using the last frame as current frame.
    '''
    if last_contour_centers is not None:
        for i in range(len(contour_centers)):
            dis = np.sum(np.abs(contour_centers[i]-last_contour_centers[i]))
            # print('max_dis:', dis)
            if dis > max_dis:
                
                contour_centers = (1-smooth_factor)*last_contour_centers+smooth_factor*contour_centers
                break
    return contour_centers



def PlotCenters(centers, cimg):
    ''' plot the centers of pins on colored images '''
    for i in range(len(centers)):
        round_point=np.round(centers[i]).astype("int") 
        cv2.circle(cimg,( round_point[0], round_point[1]),1,(0,255,0),2)  # draw centers

    return cimg

def PlotPoint(point, cimg):
    ''' plot a single point on colored image '''
    round_point=np.round(point).astype("int") # blob positions
    cv2.circle(cimg,( round_point[0], round_point[1]),3,(255,255,155),2)  # draw centers, radius, thickness

    return cimg

def PlotDisplacement(original_centers, centers, idx):
    ''' plot the displacement with vector field graph '''
    save_path = './img_displacement3/'
    plt.figure(figsize=(7.5,6))
    
    ax = plt.gca()
    X=original_centers[:, 0]
    Y=original_centers[:, 1]
    U=(centers-original_centers)[:,0]  # x-displacements
    V=(centers-original_centers)[:,1]  # y-displacements

    Z = np.sqrt(U*U + V*V)
    xi = np.linspace(0,256,100)
    yi = np.linspace(0,256,100)
    # grid the data: transform the iregular data points to regular grid data points
    from scipy.interpolate import griddata
    # zi = griddata(X,Y,Z,xi,yi, interp='linear')
    zi = griddata((X, Y), Z, (xi[None,:], yi[:,None]), method='cubic')  # method: linear, cubic, nearest
    plt.contourf(xi, yi, zi, 50, vmin=0, vmax=30)  # 2d color map: 50 is the continuous extent, vmin and vmax should match the smallest and largest value to make color look normal

    ax.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1, width=0.003)  # plot the displacement arrows
    ax.set_xlim([0, 256])
    ax.set_ylim([256, 0])  # align with the image axis order
    plt.draw()
    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')
    plt.colorbar() # draw colorbar

    # plt.show()
    plt.savefig(save_path+str(idx)+'.png')



def Norm(contour_centers):
    central_pos=np.average(contour_centers, axis=0)
    p=(np.array(contour_centers)-central_pos).tolist()
    select_pos= p[8:14]+p[16:23]+p[25:33]+p[35:44]+p[46:56]+p[58:69]+p[71:81]+p[83:92]+p[94:102]+p[104:111]+p[113:119] # select 91
    average_mag=np.average(np.abs(select_pos))
    norm_pos=select_pos/average_mag
    # np.save('real_pos', norm_pos)

    return norm_pos



if __name__ == '__main__':
    
    path = './img3/0002.png'
    save_path = './img_detected_new3/new.png'
    NUM_PINS=127
    SELECT_NUM_PINS=91 # drop the most outside circle of pins  91 or 61 
    # img_width=256
    # img_height=256
    pins_x=[]
    pins_y=[]
    last_valid_x=[]
    last_valid_y=[]
    last_contour_centers=None
    num_images = 0

    time0=time.time()
    img = cv2.imread(path,0)
    cimg, edge_detected_image, contour_centers = ImageProcessing(img)

    cimg, contour_centers = ContourCenterCheck(contour_centers, cimg, NUM_PINS=NUM_PINS)

    contour_centers=CenterRegister(contour_centers, cimg)
    print(len(contour_centers))
    central_pos=np.average(contour_centers, axis=0)
    print(central_pos)
    p=(np.array(contour_centers)-central_pos).tolist()
    select_pos= p[8:14]+p[16:23]+p[25:33]+p[35:44]+p[46:56]+p[58:69]+p[71:81]+p[83:92]+p[94:102]+p[104:111]+p[113:119] # select 91
    average_mag=np.average(np.abs(select_pos))
    print(average_mag)
    norm_pos=select_pos/average_mag
    np.save('real_pos', norm_pos)
    plot_list_new(norm_pos)


    
    cimg = PlotCenters(contour_centers, cimg)
    cv2.imwrite(save_path,cimg)

    time3 = time.time()
    print('time: {:4f}, {:4f} ,{:4f}'.format(time1-time0, time2-time1, time3-time2))
