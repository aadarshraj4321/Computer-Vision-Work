import numpy as np
import cv2
import dlib
import face_recognition


cap = cv2.VideoCapture("tony.mp4")
all_face_locations = []

while True:
    ret,frame = cap.read()
    resize_frame = cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
    all_face_locations = face_recognition.face_locations(resize_frame,model="hog")

    for index,current_face_location in enumerate(all_face_locations):
        top_pos,right_pos,bottom_pos,left_pos = current_face_location
        top_pos *= 4
        right_pos *= 4
        bottom_pos *= 4
        left_pos *= 4
        current_face_image = frame[top_pos:bottom_pos,left_pos:right_pos]
        current_face_image = cv2.GaussianBlur(current_face_image,(155,155),30)
        frame[top_pos:bottom_pos,left_pos:right_pos] = current_face_image
    cv2.imshow("Blur Face",frame)
    if cv2.waitKey(1) & 0xff == ord("f"):
        break

cap.release()
cv2.destroyAllWindows()


