# This code attempts to recognize faces from an image
# Code from https://github.com/Rob-Johnson/opencv-eigenfaces/blob/master/test_eigenfaces.py

import cv2
import numpy
import random

def read_csv(filename='faces.csv'):
	csv = open(filename, 'r')
	return csv 

def create_and_train_model_from_dict(label_matrix):
	# Create and train eigenface model
	model = cv2.createEigenFaceRecognizer()
	images = label_matrix.values()
	labels = numpy.array(label_matrix.keys())
	#model.train()
	model.train(images, labels)
	return model 

def predict_image_from_model(model, image):
	# Given an eigenface model, predict the label of an image
	prediction = model.predict(image)
	return prediction

def prepare_training_testing_data(file):
    # prepare testing and training data from file
    lines = file.readlines()
    training_data, testing_data = split_test_training_data(lines)
    return training_data, testing_data

def split_test_training_data(data, amt=9):
    # Split a list of image files: amt training, the rest testing
    random.shuffle(data)
    training_data = data[amt:]
    testing_data = data[:amt]
    return training_data, testing_data

def create_label_matrix_dict(input_file):
	# Create a dict of "label" -> "matrix" from file
	# In this instanc, "matrix" is an image
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

def read_matrix_from_file(filename):
	file_matrix = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
	return file_matrix

if __name__ == '__main__':
	training_data, testing_data = prepare_training_testing_data(read_csv())
	data_dict = create_label_matrix_dict(training_data)
	model = create_and_train_model_from_dict(data_dict)

	for line in testing_data:
		filename, label = line.strip().split(';')
		predicted_label = predict_image_from_model(model, read_matrix_from_file(filename))
		print('Predicted: {}; Actual: {}'.format(predicted_label[0], label))


