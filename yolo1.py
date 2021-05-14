import numpy as np
import cv2




#Load YOLO
net = cv2.dnn.readNet("weights/yolov3.weights","cfg/yolov3.cfg")
classes = []
with open("coco.names","r") as f:
    classes = [line.strip() for line in f.readlines()]




#print(classes)




layer_names = net.getLayerNames()
outputlayers = [layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0,255,size=(len(classes),3))




cap = cv2.VideoCapture(0)




while True:
    _,frame = cap.read()








    height, width, channels = frame.shape




    #Detecting Objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416,416),(0,0,0), True, crop=False)
    #for b in blob:
        #for n, img_blob in enumerate(b):
            #cv2.imshow(str(n), img_blob)












    net.setInput(blob)
    outs = net.forward(outputlayers)
    #print(outs)




    #Showing Information on the Screen




    class_ids=[]
    confidences = []
    boxes = []




    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.1 :
                #Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1]* height)
                w = int(detection[2]* width)
                h = int(detection[3]*height)




                #cv2.circle(img, (center_x, center_y), 10, (0,255,0), 2)




                #Rectangle Coordinates
                x = int(center_x - w /2)
                y = int(center_y - h /2)




                #cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                boxes.append([x,y,w,h])
                confidences.append(float(confidence))
                class_ids.append(class_id)




    print(len(boxes))
    #objects_detected = len(boxes)




    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    #print(indexes)
    font = cv2.FONT_HERSHEY_PLAIN




    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[i]
            cv2.rectangle(frame, (x,y), (x+w, y+h), color,2)
            cv2.putText(frame, label, (x, y+30), font, 1, color, 3)
            print(label)




    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()