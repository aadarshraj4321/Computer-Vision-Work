import cv2
import dlib
import numpy as np
import face_recognition

img = cv2.imread("andrew.jpg")
all_face_location = face_recognition.face_locations(img,number_of_times_to_upsample=2, model="hog")

# Slice the face from image

for index, current_face_recogniton in enumerate(all_face_location):
    top_pos, right_pos, bottom_pos, left_pos = current_face_recogniton
    print("Found face {} at top:{}, right:{}, bottom:{}, left:{}".format(index, top_pos, right_pos, bottom_pos, left_pos))
    cv2.rectangle(img,(left_pos, top_pos), (right_pos, bottom_pos), (0,0,255), 2)
new_img = img[top_pos:bottom_pos, left_pos:right_pos]

cv2.imshow("img",img)
cv2.imshow("new_img",new_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
