import cv2
import numpy as np


FACE_POR = 10.0/100
SPAN = 70

def build_eraser(img,face):
    (x,y,w,h) = face
    my_img = fit_face(w,h)
    
    mask = np.zeros((w,h),dtype=np.int)
    
    w_until = int(FACE_POR*w)
    h_until = int(FACE_POR*h)

    cnt_table = []
    rgb_table = []
    
    delta_x = w/2 - w_until/2
    delta_y = h/2 - h_until/2
    
    a = x+int(delta_x)
    b = x+int(delta_x) + w_until

    c = y + int(delta_y) 
    d = y + int(delta_y) + h_until

    for i in range(a,b):           # x-axis
        for j in range(c,d):       # y-axis
            (a,b,c) = img[j,i]
            if (a,b,c) in rgb_table:
                index = rgb_table.index((a,b,c))
                cnt_table[index] = cnt_table[index]+1
            else:
                rgb_table.append((a,b,c))
                cnt_table.append(1)

    m= max(cnt_table)

    index = cnt_table.index(m)
    (a1,b1,c1) = rgb_table[index]

    for i in range(w):
        for j in range(h):
            (a,b,c) = img[y+j,x+i]
            if (a1-SPAN < a) & (a < a1+SPAN):
                if (b1-SPAN < b) & (b < b1+SPAN):
                    if (c1-SPAN < c) & (c < c1+SPAN):
                        img[y+j,x+i] = my_img[j,i]
                        #img[y+j,x+i] = [0,0,0]

    cv2.imshow("Faces found",img)
    cv2.waitKey(0)
   


def fit_face(w,h):
    img = cv2.imread('garam.png')
    return cv2.resize(img,(w,h))

    
    

def face_detect(imagePath):
    cascPath = 'haarcascade_frontalface_default.xml'
    
    # Create the haar cascade
    faceCascade = cv2.CascadeClassifier(cascPath)

    # Read the image
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv2.cv.CV_HAAR_SCALE_IMAGE
        )

    return image,faces[0]

    # print "Found {0} faces!".format(len(faces))

    # # Draw a rectangle around the faces
    # for (x, y, w, h) in faces:
    #     cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # cv2.imshow("Faces found", image)
    # cv2.waitKey(0)

img,face = face_detect('hani.jpg')
build_eraser(img,face)
