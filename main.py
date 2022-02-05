
# Sanskar Goel The Sparks Foundation - IoT and Computer Vision Internship
# Task 1 : Object Detection

import cv2  # Importing the required library file
img=cv2.imread('Test.png')
classNames= []
classFile = 'coco.names'  #Incorporating the dataset holding different classes of objects
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'  #Incorporating the classifier
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)



classIds, confs, bbox = net.detect(img, confThreshold=0.5)
print(classIds,bbox)
for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
        cv2.rectangle(img, box, color=(0, 0, 255), thickness=3)   #Formation of the box around objects
        cv2.putText(img, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

cv2.imshow("Output", img)
cv2.waitKey(0)