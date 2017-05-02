# All of our raw images are of different sizes. We need to 
# standardize them.
# Images should be in the .jpg format, and of size (256, 256)
# Some images are rotated because opencv doesn't preserve
# orientation data from phones. 

import numpy
import cv2 

# face detector classifier
CASCADE = './data/haarcascade_frontalface_default.xml'


# Filepaths for input and output
BASEINNAME = 'images/raw/michael/img{}.jpg'
BASEOUTNAME = 'images/processed/michael/{}pic{}.jpg'

# Number of pictures in the file
NUMPICS = 12

# Given a cascade detector, use it on all of the images in a folder
# Images must be named in a certain format (see BASEINNAME above)
# Stores all faces detected in a different folder (see BASEOUTNAME above)
def detect_faces(face_cascade):
	# For each picture...
	for i in range(1, NUMPICS+1):
		print "Working on {}".format(i)
		filename = BASEINNAME.format(i)

		# Read in the image
		img = cv2.imread(filename)

		# Rotate if necessary
		img = numpy.rot90(img, 1)
		minSize = min(img.shape[0], img.shape[1])/10

		# Convert to greyscale
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		# Detect faces in the image - there should only be one
		faces = face_cascade.detectMultiScale(gray, scaleFactor=1.01, minNeighbors=5, minSize=(minSize, minSize), flags=cv2.CASCADE_SCALE_IMAGE)

		print "{} faces detected".format(len(faces))

		# For each face detected
		iteration=0
		for (x,y,w,h) in faces:
			iteration += 1
			# roi = region of interest
			roi = img[y:y+h, x:x+w]

			# Save that region of interest as a 256x256 image
			imgR = cv2.resize(roi, (256, 256))
			cv2.imwrite(BASEOUTNAME.format(i, iteration), imgR)

if __name__ == '__main__':
	# Create the detector
	face_cascade = cv2.CascadeClassifier(CASCADE)
	detect_faces(face_cascade)

		