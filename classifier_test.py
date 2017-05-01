# This program tests which cascade classifier is most efficient
# Originally from http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_objdetect/py_face_detection/py_face_detection.html

import numpy
import cv2
import sys

# CHANGE THIS to your image's filepath
INNAME = 'images/raw/puneeth/img8.jpg'

cascade_default = cv2.CascadeClassifier('./data/haarcascade_frontalface_default.xml')
cascade_alt1 = cv2.CascadeClassifier('./data/haarcascade_frontalface_alt1.xml')
cascade_alt2 = cv2.CascadeClassifier('./data/haarcascade_frontalface_alt2.xml')
cascade_tree = cv2.CascadeClassifier('./data/haarcascade_frontalface_tree.xml')

def rotate_image(image):
	rotated = numpy.rot90(image, 3)
	return rotated

     
img = cv2.imread(INNAME)
print(img.shape)
sys.exit(0)
# imgd = cv2.imread(INNAME)
# imga1 = cv2.imread(INNAME)
# imga2 = cv2.imread(INNAME)
# imgt = cv2.imread(INNAME)

longer = max(img.shape[0], img.shape[1])
shorter = min(img.shape[0], img.shape[1])

ratio = img.shape[1]/img.shape[0]

new_dims = (512, int(512*ratio))
imgd = cv2.resize(img, new_dims, interpolation=cv2.INTER_AREA)
imga1 = cv2.resize(img, new_dims, interpolation=cv2.INTER_AREA)
imga2 = cv2.resize(img, new_dims, interpolation=cv2.INTER_AREA)
imgt = cv2.resize(img, new_dims, interpolation=cv2.INTER_AREA)

minSize = min(imgd.shape[0], imgd.shape[1])/50
gray = cv2.cvtColor(imgd, cv2.COLOR_BGR2GRAY)

print("Loaded image and classfiers")

faces_d = cascade_default.detectMultiScale(gray, scaleFactor=1.01, minNeighbors=5, minSize=(minSize, minSize), flags=cv2.CASCADE_SCALE_IMAGE)

print("Cascaded with default")
print("Found {} faces".format(len(faces_d)))

faces_a1 = cascade_alt1.detectMultiScale(gray, scaleFactor=1.01, minNeighbors=5, minSize=(minSize, minSize), flags=cv2.CASCADE_SCALE_IMAGE)

print("Cascaded with alt1")
print("Found {} faces".format(len(faces_a1)))

faces_a2 = cascade_alt2.detectMultiScale(gray, scaleFactor=1.01, minNeighbors=5, minSize=(minSize, minSize), flags=cv2.CASCADE_SCALE_IMAGE)

print("Cascaded with alt2")
print("Found {} faces".format(len(faces_a2)))

faces_tree = cascade_tree.detectMultiScale(gray, scaleFactor=1.01, minNeighbors=5, minSize=(minSize, minSize), flags=cv2.CASCADE_SCALE_IMAGE)

print("Cascaded with tree")
print("Found {} faces".format(len(faces_tree)))

for (x,y,w,h) in faces_d:
    cv2.rectangle(imgd,(x,y),(x+w,y+h),(255,0,0),2)
    
cv2.imshow("default", imgd)
cv2.waitKey(0)

for (x,y,w,h) in faces_a1:
    cv2.rectangle(imga1,(x,y),(x+w,y+h),(255,0,0),2)
    
cv2.imshow("alt1", imga1)
cv2.waitKey(0)

for (x,y,w,h) in faces_a2:
    cv2.rectangle(imga2,(x,y),(x+w,y+h),(255,0,0),2)
    
cv2.imshow("alt2", imga2)
cv2.waitKey(0)

for (x,y,w,h) in faces_tree:
    cv2.rectangle(imgt,(x,y),(x+w,y+h),(255,0,0),2)
    
cv2.imshow("tree", imgt)
cv2.waitKey(0)
