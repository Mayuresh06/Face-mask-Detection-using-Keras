from keras.preprocessing import image
import h5py
from keras.models import load_model
import numpy as np
import cv2
from mtcnn.mtcnn import MTCNN
model=load_model("face_mask_detection.h5")
detector = MTCNN()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS,1)

while(True):
    ret,frame=cap.read()
    
    face_cord = detector.detect_faces(frame)
    if face_cord != []:
        for person in face_cord:
            bounding_box = person['box']
            keypoints = person['keypoints']
            face_image=frame[bounding_box[1]:(bounding_box[1] + bounding_box[3]),bounding_box[0]:(bounding_box[0]+bounding_box[2])]
            cv2.rectangle(frame,
                         (bounding_box[0], bounding_box[1]),
                          (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
                          (0,155,255),
                          2)
    
            gray = cv2.cvtColor(face_image,cv2.COLOR_BGR2GRAY)
            resize = cv2.resize(gray,(255,255))
            img_pred = image.img_to_array(resize)
            img_pred = np.expand_dims(img_pred, axis = 0)
            result = model.predict(img_pred)
    
            for i in result:
                index = np.argmax(i)
                print(index)
                if index == 0:
                    text1 = "without_mask"
                elif index == 1:
                    text1 = "with_incorrect_mask"
                elif index==2:
                    text1 = "with_mask"
    
            font=cv2.FONT_HERSHEY_COMPLEX_SMALL
            text = 'output: ' + str(text1)
            frame=cv2.putText(frame,text,(10,50),font,1,(0,255,0))
    
    cv2.imshow('capture',frame)
    k=cv2.waitKey(5)
    if k==27:
        cv2.destroyAllWindows()
        break
        
cap.release()
cv2.destroyAllWindows()
