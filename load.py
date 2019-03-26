# -*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from keras.models import load_model
import numpy as np
from keras.utils import to_categorical
import os
from skimage.io import imsave, imread
from skimage.transform import resize, rotate
from keras.optimizers import SGD, Adam
import matplotlib.pyplot as plt
from os_cleanup import get_keras_variables
import pickle

def load_model(model_path, weights_path):
	# load json and create model
	json_file = open(model_path, 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	loaded_model.load_weights(weights_path)
	print("Loaded model from disk")

	loaded_model.compile(loss='binary_crossentropy',
	              optimizer='SGD',
	              metrics=['accuracy'])
	# loaded_model.summary()
	return loaded_model
	
def load_image(img_path):
	save_path = "../bin/gui/" + 'temp.png'
	img = imread(img_path)
	img_resized = resize(img, (300,300)) 
	imsave(save_path, img_resized)
	return save_path

def draw_plot(classes, prediction):
	save_path = "../bin/gui/" + 'results.png'
	fig, ax = plt.subplots()

	# Example data
	y_pos = np.arange(len(classes))
	performance = prediction


	ax.barh(y_pos, performance, align='center',
	        color='blue', ecolor='black')
	ax.set_yticks(y_pos)
	ax.set_yticklabels(classes)
	plt.yticks(size = 10)
	plt.xticks(size = 10)
	ax.invert_yaxis()  # labels read top-to-bottom
	ax.set_xlabel(u'Pewnosć %', size=10)
	ax.set_title('Wyniki')
	for i, v in enumerate(performance):
		print(v)
		if i == 0:
			temp = -10
		else:
			temp = 3
		if v == 0:
			v = 0
		ax.text(v+temp, i , str(v), color='black', fontweight='bold', size=20)
	fig.savefig(save_path)


	img = imread(save_path)
	img_resized = resize(img, (400,500)) 
	imsave(save_path, img_resized)
	return save_path

	
def predict(paths):
	print(paths)
	# exit()
	model = load_model(paths['model_path'], paths['weights_path'])
	# model = load_model('../bin/incepV3/model.json', '../bin/incepV3/weights.h5')
	shape = (int(model.input.shape[1]), int(model.input.shape[2]))
	img = imread(paths['img_path'])
	img = resize(img, (shape)) 
	img = img.astype('float32')
	print(img.shape)
	classes = ('NON_TUMOR', 'SOLID', 'CRIBRIFORM', 'ACINAR', 'MICROPAP')
	x = np.ndarray(shape=(1, img.shape[0], img.shape[1], img.shape[2]))
	x[0,] = img
	prediction = model.predict(x)
	prediction = list(prediction[0])
	print(prediction)
	mapped_results = {}
	prediction = [round(i *100, 2) for i in prediction]
	for i,cancer_type in enumerate(classes):
		mapped_results[prediction[i]] = cancer_type
	print(prediction)
	print(mapped_results)
	sorted_results = sorted(prediction, reverse=True)
	print(sorted_results)
	classes = []
	for result in sorted_results:
		classes.append(mapped_results[result])
	print(classes)
	return draw_plot(classes, sorted_results)

def lab():
	# exit()
	name =  'incepV3'
	# model = load_model('../bin/'+name+'/model.json', '../bin/'+name+'/weights.h5')
	# shape = (int(model.input.shape[1]), int(model.input.shape[2]))
	partition, labels, labels_string = get_keras_variables()
	# dir_path = '/media/santis/5800-7A46/Histo_img_notordered'
	total = len(partition['test'])
	# x = np.ndarray(shape=(total, shape[0], shape[1], 3))
	y = np.ndarray(shape=(total, 1))
	# # print(total)
	for i, image in enumerate(partition['test']):
	# 	img_path = dir_path + '/' + image + '.tif'
	# 	print(img_path)
	# 	img = imread(img_path)
	# 	img = resize(img, (shape)) 
	# 	img = img.astype('float32')
	# 	x[i,] = img
		y[i] = labels[image]
	# history = model.predict(x)
	# f = open('../bin/test_imges.pckl', 'wb')
	# pickle.dump(history, f)
	# f.close()
	# print('saved')
	f = open('../bin/test_imges.pckl', 'rb')
	history = pickle.load(f)
	f.close()

	history = (history == history.max(axis=1, keepdims=1)).astype(float)
	# print(history)
	ground_truth = np.ndarray(shape=(total, 5))
	ground_truth = to_categorical(y, 5)
	# print(ground_truth)

	# mul = np.multiply(history, ground_truth)
	mul = history * ground_truth
	# print(mul)
	mul = mul[:,2]

	print((mul==1).sum())

	exit()


# lab()