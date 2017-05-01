# All of our raw images are of different sizes. We need to standardize them somehow

import numpy
import cv2 

CASCADE = './data/haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(CASCADE)


BASEINNAME = 'images/raw/puneeth/img{}.jpg'
BASEOUTNAME = 'images/processed/puneeth/{}pic{}.jpg'
NUMPICS = 10

for i in range(12, 17):
	print "Working on {}".format(i)
	filename = BASEINNAME.format(i)

	img = cv2.imread(filename)
	img = numpy.rot90(img, 2)
	minSize = min(img.shape[0], img.shape[1])/10

	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	faces = face_cascade.detectMultiScale(gray, scaleFactor=1.01, minNeighbors=5, minSize=(minSize, minSize), flags=cv2.CASCADE_SCALE_IMAGE)

	iteration=1
	for (x,y,w,h) in faces:
		print "{} faces detected".format(len(faces))
		roi = img[y:y+h, x:x+w]

		imgR = cv2.resize(roi, (256, 256))
		cv2.imwrite(BASEOUTNAME.format(i, iteration), imgR)
		iteration+= 1