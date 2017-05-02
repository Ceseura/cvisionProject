import cv2
import sys
import numpy

def display_image(PATH):
	image = cv2.imread(PATH)
	#image = numpy.rot90(image, 1)

	cv2.imshow('', image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def perm_rotate_image(PATH):
	image = cv2.imread(PATH)
	print(image.shape)
	image = numpy.rot90(image, 3)
	#cv2.imshow('', image)
	#cv2.waitKey(0)
	cv2.imwrite("images/raw/utilsOutput.jpg", image)

PATH = ''
NAME = "pic{}.jpg"

if __name__ == '__main__':

	if len(sys.argv) is not 3:
		print("try: 'python testOrientation.py <filepath> <show/rotate>'")
		sys.exit(1)

	PATH = sys.argv[1]

	# Show displays all of the images in the specified folder with 
	# name formatted like: imgX.jpg
	if sys.argv[2] == 'show':
		#display_image(PATH)
		for i in range(1, 16):
			filename = PATH + NAME.format(i)
			print "Working on {}".format(filename)
			display_image(filename)

	# Rotate rotates one image
	elif sys.argv[2] == 'rotate':
		perm_rotate_image(PATH)


