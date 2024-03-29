# This code trains a recognizer
# Originally from https://github.com/Rob-Johnson/opencv-eigenfaces/blob/master/test_eigenfaces.py
# But edited

import cv2
import sys
import random
import numpy

# Takes filename (string) as input, outputs a file object
def read_csv(filename):
	csv = open(filename, 'r')
	return csv 

# Takes a file object as input, returns a list of filename:label (string)
def prepare_training_data(fileData):
    # prepare testing and training data from file
    #lines = file.readlines()
    training_data = shuffle_training_data(fileData)
    return training_data

# Takes a list of filename:label (string), returns a shuffled list of filename:label (string)
def shuffle_training_data(data):
    # Randomly shuffle the training data
    random.shuffle(data)
    return data

# Takes a list of filename:label (string) and returns a dictionary of {label:matrix_from_filename}
def create_label_matrix_dict(input_file):
	# Create a dict of "label" -> "matrix" from file
	# In this instance, "matrix" is an image
	label_dict = {}

	for line in input_file:
		# Separate each line on the ';' marker
		filename, label = line.strip().split(';')

		# Update current key if it exists, else make a new key
		if int(label) in label_dict:
			current_files = label_dict.get(label)
			numpy.append(current_files, read_matrix_from_file(filename))
		else:
			label_dict[int(label)] = read_matrix_from_file(filename)

	return label_dict

# Takes a filename (string) and returns a matrix representing the image
# with normalized intensity
def read_matrix_from_file(filename):
	file_matrix_temp = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
	file_matrix = cv2.equalizeHist(file_matrix_temp)
	return file_matrix

# Takes a dictionary of {label:matrix_image} and creates an eigenface model
def create_and_train_model_from_dict(label_matrix, type):
	# Create and train eigenface model
	if type == 'lbph':
		model = cv2.createLBPHFaceRecognizer()
	elif type == 'eigen':
		model = cv2.createEigenFaceRecognizer()
	elif type == 'fisher':
		model = cv2.createFisherFaceRecognizer()
	images = label_matrix.values()
	labels = numpy.array(label_matrix.keys())
	model.train(images, labels)
	return model 

MODELNAME = './models/model{}{}.xml'
NAME = {'eigen':'E', 'fisher':'F', 'lbph':'L'}
if __name__ == '__main__':

	if len(sys.argv) is not 4:
		# <csv_of_images> should be generated by 'create_csv.py'
		# <model_type> should be 'eigen', 'fisher', or 'lbph'
		# <how_many> should be a number indicating how many models
		#         to be trained
		print("Invalid arguments: try 'python trainRecognizer.py <csv_of_images> <model_type> <how_many>'")
		sys.exit(1)

	# Open the CSV
	CSVFILENAME = sys.argv[1]
	csv = read_csv(CSVFILENAME)
	fileData = csv.readlines()

	# Iterate <how_many> times
	for i in range(int(sys.argv[3])):

		training_data = prepare_training_data(fileData)
		data_dict = create_label_matrix_dict(training_data)
		model = create_and_train_model_from_dict(data_dict, sys.argv[2])
		mName = MODELNAME.format(NAME[sys.argv[2]], i+1)
		model.save(mName)
		print "trained model {}".format(i+1)


