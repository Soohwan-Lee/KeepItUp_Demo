import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = '0' # 여러개 사용시 '0,1,2' 식으로 하나의 문자열에 입력
gpus = tf.config.experimental.list_physical_devices('GPU') # 호스트 러나임에 표시되는 GPU 장치 목록 반환

if gpus: # 반환된 GPU 장치 목록이 있다면
    try: # 해당 장치에 대한 메모리 증가 활성화 여부 설정
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e: # try문 실패시에 에러문구 출력
        print("Error!")

import tensorflow_hub as hub
import cv2
from matplotlib import pyplot as plt
from matplotlib import animation
import numpy as np
import pandas as pd
import math
import socket
import time

### Load Model
# model = hub.load('https://tfhub.dev/google/movenet/multipose/lightning/1')
model = tf.saved_model.load('./model/moveNet/')
movenet = model.signatures['serving_default']

# ### Load Video
video_path = './data/videos/'
actionThree = video_path + 'actionPracticeWhite.mp4'

### Draw EDGES
EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}

### Vector List
vectorList = [
    [0,1],
    [0,2],
    [1,3],
    [2,4],
    [3,5],
    [0,6],
    [1,7],
    [6,7],
    [6,8],
    [7,9],
    [8,10],
    [9,10]
]

### Color for each person
color_mapping = {
    0: (0, 0, 255),
    1: (0, 255, 0),
    2: (0, 255, 255)
}

# Function to loop through each person detected and render
def loop_through_people(frame, keypoints_with_scores, edges, confidence_threshold, movingArray):
    for index, person in enumerate(keypoints_with_scores):
        draw_connections(frame, person, edges, confidence_threshold, index, movingArray)
        draw_keypoints(frame, person, confidence_threshold, index, movingArray)


def draw_keypoints(frame, keypoints, confidence_threshold, index, movingArray):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    movingArray = np.multiply(movingArray, x)
    index = index * 2
    
    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx + movingArray[index]), int(ky)), 3, color_mapping.get(index/2), -1)
            cv2.circle(frame, (int(kx + movingArray[index + 1]), int(ky)), 3, color_mapping.get(index/2), -1)


def draw_connections(frame, keypoints, edges, confidence_threshold, index, movingArray):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    movingArray = np.multiply(movingArray, x)
    index = index * 2
    
    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]
        
        if (c1 > confidence_threshold) & (c2 > confidence_threshold):      
            cv2.line(frame, (int(x1 + movingArray[index]), int(y1)), (int(x2 + movingArray[index]), int(y2)), color_mapping.get(index/2), 2) # color[index]
            cv2.line(frame, (int(x1 + movingArray[index + 1]), int(y1)), (int(x2 + movingArray[index + 1]), int(y2)), color_mapping.get(index/2), 2)




### Variables for Calculating FPS
prevTime = time.time() # previous time

if __name__ == "__main__":
    ### Variables
    numberOfPeople = 3

    ### Loading Video File
    cap = cv2.VideoCapture(actionThree)    ### Change the File Here!!
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)

    
    while cap.isOpened():
        ret, frame = cap.read()
        


        ### Variables for each frame
        initialTime = time.time()
        
        # Resize image
        img = frame.copy()
        img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 384,640)
        # img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 128, 256)

        input_img = tf.cast(img, dtype=tf.int32)

        # frame = cv2.resize(frame, (1280, 640))
        
        
        # Detection section
        results = movenet(input_img)
        
        # Get the keypoints_with_score
        keypoints_with_scores = results['output_0'].numpy()[:,:,:51].reshape((6,17,3))
        keypoints_with_scores = keypoints_with_scores[:numberOfPeople]
        
        # Sort with each person from left to right
        sorted_indices = np.argsort(keypoints_with_scores[:, 0, 1])
        keypoints_with_scores = keypoints_with_scores[sorted_indices]
        
        # Calculate the moving value
        array = [keypoints_with_scores[0][0][1], keypoints_with_scores[1][0][1], keypoints_with_scores[2][0][1]]
        movingArray = [array[j] - array[i] for i in range(len(array)) for j in range(len(array)) if i != j]

        
        keypoints_only = np.delete(keypoints_with_scores,2,2)
        keypoints_only_body = np.delete(keypoints_only, [0,1,2,3,4], 1)
        
    

        # Render keypoints 
        loop_through_people(frame, keypoints_with_scores, EDGES, 0.1, movingArray)
#         loop_through_people(frame, [keypoints_with_scores[0]], EDGES, 0.1)    # Check for first person.....
        
        ### Calculate & Print FPS
        # Count Frame
        curTime = time.time()	# current time
        fps = 1 / (curTime - prevTime)
        prevTime = curTime
        # Save FPS
        fps_str = "FPS : %0.1f" %fps
        # FPS print
        cv2.putText(frame, fps_str, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        
        cv2.imshow('Movenet Multipose', frame)
        
        
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()