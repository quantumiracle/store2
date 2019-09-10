import numpy as np
import cv2
import imutils
from localize import ImageProcessing, ContourCenterCheck, CenterRegister, PointCheck, Norm
from train_with_data import Classifier
from deform_visualize import plot_list_new
from train_policy.td3_multiprocess_all import TD3_Trainer, ReplayBuffer

cap = cv2.VideoCapture(1)

def preprocessing(image):
    ''' rotate and crop and resize '''
    height=frame.shape[0]
    width=frame.shape[1]
    # print(ret, height, width)
    if ret:
        # rotate the image to standard direction
        angle=30   # -13
        rotated = imutils.rotate(frame, angle)

        crop_img = rotated[:, int((width-height)/2):int((width+height)/2)]  # crop to be square
        resized = cv2.resize(crop_img, (256, 256), interpolation = cv2.INTER_AREA)

    # Display the resulting frame
    # cv2.imshow('frame', resized)

    return resized[:,:, 0]  # only 1 channel

def InitPredict(model_path = './model/class_obj'): 
    obs_dim = 182  # total 280: 0 object index, 1-3 rotation value, 4-6 average contact point position, 7-279 pins positions
    state_dim = 5  # 3 rotation, 2 position
    lr=2e-2
    num_obj=3  # number of objects
    classifier = Classifier(obs_dim, state_dim, lr, num_obj)
    classifier.load(model_path) 
    return classifier

def RLpolicy(state):
    replay_buffer = ReplayBuffer(1e6)
    td3_trainer=TD3_Trainer(replay_buffer)
    td3_trainer.load_model(model_path)

    action = td3_trainer.policy_net.get_action(state)
    return action

 

cnt=0

classifier=InitPredict()
while(True):
    NUM_PINS=127
    SELECT_NUM_PINS=91 # drop the most outside circle of pins  91 or 61 
    last_contour_centers=None
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    img=preprocessing(frame)

    cimg, edge_detected_image, contour_centers = ImageProcessing(img)

    cimg, contour_centers = ContourCenterCheck(contour_centers, cimg, NUM_PINS=NUM_PINS)

    contour_centers=CenterRegister(contour_centers, cimg)

    contour_centers = PointCheck(contour_centers, last_contour_centers, max_dis=15)  # larger than 15 is mis-registered
    if cnt>=10:
        last_contour_centers = contour_centers
    
    norm_pos=Norm(contour_centers).reshape(-1)  # (x,y,x,y...)
    # script_action=RLpolicy(norm_pos)
    # real_sensor_size=
    # sim_sensor_size=3.9
    # sim_action_unit=0.02
    # sim_action=script_action*sim_action_unit
    # action=sim_action*real_sensor_size/sim_sensor_size

   cnt+=1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()