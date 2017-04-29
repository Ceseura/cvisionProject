# All of our raw images are of different sizes. We need to standardize them somehow

import numpy
import cv2 

face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
CASCADE = './haarcascade_frontalface_default.xml'

BASEINNAME = 'images/raw/alex/img{}.jpg'
BASEOUTNAME = 'images/raw/alex/{}pic{}.jpg'
NUMPICS = 11

for i in range(1, NUMPICS+1):
	print "Working on {}".format(i)
	filename = BASEINNAME.format(i)

	img = cv2.imread(filename)
	minSize = min(img.shape[0], img.shape[1])/10

	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	faces = face_cascade.detectMultiScale(gray, scaleFactor=1.01, minNeighbors=5, minSize=(minSize, minSize), flags=cv2.CASCADE_SCALE_IMAGE)
	iteration=1
	for (x,y,w,h) in faces:
		print "{} faces detected".format(len(faces))
		roi = img[y:y+h, x:x+w]

		cv2.imwrite(BASEOUTNAME.format(i, iteration), roi)
		iteration+= 1