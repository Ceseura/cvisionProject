# All of our raw images are of different sizes. We need to 
# standardize them.
# Images should be in the .jpg format, and of size (256, 256)
# Some images are rotated because opencv doesn't preserve
# orientation data from phones. 

import numpy
import cv2 

# face detector
CASCADE = './data/haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(CASCADE)

# Filepaths for input and output
BASEINNAME = 'images/raw/puneeth/img{}.jpg'
BASEOUTNAME = 'images/processed/puneeth/{}pic{}.jpg'
# Number of pictures in the file
NUMPICS = 10

# For each picture...
for i in range(12, 17):
	print "Working on {}".format(i)
	filename = BASEINNAME.format(i)

	# Read in the image
	img = cv2.imread(filename)
	# Rotate if necessary
	#img = numpy.rot90(img, 2)
	minSize = min(img.shape[0], img.shape[1])/10

	# Convert to greyscale
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# Detect faces in the image - there should only be one
	faces = face_cascade.detectMultiScale(gray, scaleFactor=1.01, minNeighbors=5, minSize=(minSize, minSize), flags=cv2.CASCADE_SCALE_IMAGE)

	# For each face detected
	iteration=1
	for (x,y,w,h) in faces:
		print "{} faces detected".format(len(faces))
		# roi = region of interest
		roi = img[y:y+h, x:x+w]

		# Save that region of interest as a 256x256 image
		imgR = cv2.resize(roi, (256, 256))
		cv2.imwrite(BASEOUTNAME.format(i, iteration), imgR)
		iteration+= 1


		