# pip install opencv-python

import cv2

cascade_face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') #catch your front face
cascade_eye = cv2.CascadeClassifier('haarcascade_eye.xml')  #detect eye
cascade_smile = cv2.CascadeClassifier('haarcascade_smile.xml')  #detect smile

cap = cv2.VideoCapture(0) # capture video using default webcam

while True:
    ret, img = cap.read()
    # img = cv2.flip(img,1,0)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # converting bgr image to gray scale
    f = cascade_face.detectMultiScale(
        g,
        scaleFactor = 1.3, # size of image
        minNeighbors = 5,
        minSize = (30,30),
    )

    for(x,y,w,h) in f:
        cv2.rectangle(img, (x,y), (x + w,y + h), (255,0,0), 3) # draw a rectangle ( image, starting, ending, color, thickness)
        gray_r = g[y : y+h,x : x+w] # extracting only gray region of face

        s = cascade_smile.detectMultiScale(
            gray_r,
            scaleFactor = 1.4,
            minNeighbors = 15,
            minSize = (20,20),
        )

        for i in s: # reading smiles
            if len(s) > 1: # if there is only 1 smile
                cv2.putText(img, "SMILING", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,255), 3, cv2.LINE_AA)

    cv2.imshow('video', img) # view image on screen
    k=cv2.waitKey(30) & 0xff

    if k==27: # press esc to quit
        break

cap.release()
cv2.destroyAllWindows()