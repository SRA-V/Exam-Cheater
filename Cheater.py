import cv2
import numpy as np
import face_recognition


cap = cv2.VideoCapture(0)
success,imgUser = cap.read()
imgUser = cv2.cvtColor(imgUser, cv2.COLOR_BGR2RGB)
encodeUser = face_recognition.face_encodings(imgUser)

cap = cv2.VideoCapture(0)
while True:
    success,img = cap.read()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)
    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
        maches = face_recognition.compare_faces(encodeUser,encodeFace)
        faceDist = face_recognition.face_distance(encodeUser,encodeFace)
        print(faceDist)
        matchIndex = np.argmin(faceDist)
        if maches[matchIndex]:
            name = "User"
            print(name)
            y1,x2,y2,x1 = faceLoc
            y1, x2, y2, x1=y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
        else:
            name = "Not User"
            print(name)
            y1,x2,y2,x1 = faceLoc
            y1, x2, y2, x1=y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
    cv2.imshow('Webcam',img)
    cv2.waitKey(1)