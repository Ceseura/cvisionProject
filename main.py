import cv2
import numpy
import sys

# Dictionary of names for display purposes
names = {'0':'ALEX', '1':'MILIN', '2':'NICO', '3': 'PUNEETH'}

# The face finder, using haar cascades
CASCADE = './data/haarcascade_frontalface_alt1.xml'
face_finder = cv2.CascadeClassifier(CASCADE)

# The face recognizer, using Eigenfaces/Fisherfaces/???
recognizer = cv2.createFisherFaceRecognizer()
recognizer.load('modelF.xml')

# Default image filepath. Can be changed if an argument is passed 
# in via terminal
IMAGEFPATH = './images/raw/alex/img5.jpg'

# If an argument has been passed in then that is the target image
if len(sys.argv) is 2:
	IMAGEFPATH = sys.argv[1]

# Reads images in normal format, greyscale, and scaled to 256x256
image = cv2.imread(IMAGEFPATH)
imageCopy = cv2.imread(IMAGEFPATH)
gray = cv2.imread(IMAGEFPATH, cv2.IMREAD_GRAYSCALE)
imageGR = cv2.resize(gray, (256, 256))
imageR = cv2.resize(image, (image.shape[1]//3, image.shape[0]//3))

# Min size is 1/10 of the smaller dimension of (height, width). 
# Anything smaller than that will be discarded. 
minSize = min(image.shape[0], image.shape[1])/10

# Display the input image, just as a sanity check
cv2.imshow("base", imageR)
cv2.waitKey(0)

# Find all of the faces in the picture
faces = face_finder.detectMultiScale(gray, scaleFactor=1.01, minNeighbors=5, minSize=(minSize, minSize), flags=cv2.CASCADE_SCALE_IMAGE)

print "{} faces detected".format(len(faces))

# For each face, mark it with a blue square, run the recognizer 
# against it, and label the square
iteration = 0
for (x, y, w, h) in faces:
	iteration += 1
	# roi = region of interest, G = greyscale, R = resized
	roi = image[y:y+h, x:x+w]
	roiG = gray[y:y+h, x:x+w]

	roiR = cv2.resize(roi, (256, 256))
	roiGR = cv2.resize(roiG, (256, 256))

	# Predict with the recognizer
	prediction = recognizer.predict(roiGR)

	print 'Predicted: {}; Confidence: {}'.format(names[str(prediction[0])], prediction[1])

	# Add rectangle and text label to faces
	cv2.putText(imageCopy, names[str(prediction[0])], (x, y), cv2.FONT_HERSHEY_PLAIN, 4, (255, 0, 0), 4)
	cv2.rectangle(imageCopy,(x,y),(x+w,y+h),(255,0,0),2)

	# Show all of the detected faces 
	cv2.imshow("face {}".format(iteration), roi)
	cv2.waitKey(0)


# Show the final picture with labels
imageCopy = cv2.resize(imageCopy, (imageCopy.shape[1]//3, imageCopy.shape[0]//3))
cv2.imshow("final", imageCopy)
cv2.waitKey(0)
cv2.destroyAllWindows()






