# -*- coding: utf-8 -*-
from os_cleanup import get_keras_variables
import numpy as np
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.utils import to_categorical
from keras.layers.core import Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, Adam, RMSprop, Adagrad
from data_generation_batches import DataGenerator
from skimage.io import imsave, imread
from skimage.transform import resize
from keras import regularizers
from keras.models import model_from_json
import os 
import matplotlib.pyplot as plt
import shutil
import pickle
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.applications.densenet import DenseNet201
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import GlobalAveragePooling2D
from keras import backend as K
from keras.utils import Sequence
from keras.callbacks import Callback




def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1.) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)




class LossHistory(Callback):
	def __init__(self): 
		self.losses = []
		self.val_losses = []
		self.acc = []
		self.val_acc = []
	def on_epoch_end(self, epoch, logs={}):
		self.losses.append(logs.get('loss'))
		self.val_losses.append(logs.get('val_loss'))

		self.acc.append(logs.get('acc'))
		self.val_acc.append(logs.get('val_acc'))
		fig = plt.figure(1)  

		 # summarize history for accuracy  
		   
		plt.subplot(211)  
		plt.plot(self.acc)  
		plt.plot(self.val_acc)  
		# plt.title(u'Dokładnosć modelu', size=10)  
		plt.ylabel(u'Dokładnosć modelu', size=15)  
		# plt.xlabel('epoch', size=10)  
		plt.legend(['train', 'test'], loc='upper left')  

		# summarize history for loss  

		plt.subplot(212)  
		plt.plot(self.losses)  
		plt.plot(self.val_losses)  
		plt.ylabel(u'Wartosć funkcji kosztu', size=15)  
		plt.xlabel(u'Epoka', size=15)  
		plt.legend(['train', 'test'], loc='upper left') 
		save_path = '/home/santis/Documents/Praca_inzynierska/bin/gui/train_results.png'
		fig.savefig(save_path)  

		img = imread(save_path)
		img_resized = resize(img, (500,500)) 
		imsave(save_path, img_resized)
		return (save_path)


def batch_generator(train_params, val_params):
	print(train_params, val_params)
	partition, labels, labels_string = get_keras_variables()

	# Generators
	train_generator = DataGenerator(partition['train'], labels, **train_params)
	validation_generator = DataGenerator(partition['validation'], labels, **val_params)

	return(train_generator, validation_generator)

def get_model(architecture, input_dim, output_dim, pretrained):
	print(architecture, input_dim, output_dim, pretrained)
	if pretrained:
		weights = 'imagenet'
	else:
		weights = None
	if architecture == 'VGG':
		model = VGG16(weights=weights, include_top=False, input_shape=(input_dim))
		if pretrained:
			for layer in model.layers:
				layer.trainable=False
		flatten = Flatten()
		dropout = Dropout(0.6)
		new_layer2 = Dense(output_dim, activation='softmax', name='my_dense_2', kernel_regularizer=regularizers.l2(0.01))
		inp2 = model.input
		out2 = new_layer2(dropout(flatten(model.output)))
		model = Model(inp2, out2)

	elif architecture == "ResNet":
		base_model = ResNet50(weights=weights, include_top=False, input_shape = (input_dim))
		# flatten = Flatten()
		if pretrained:
			for layer in base_model.layers:
				layer.trainable=False
		x = GlobalAveragePooling2D(name='avg_pool')(base_model.output)
		x = Dense(output_dim, activation='softmax', name='predictions')(x)
		model = Model(inputs=base_model.input, outputs=x)

	elif architecture == "Inception":
		base_model = InceptionV3(weights=weights, include_top=False, input_shape = (input_dim))
		# Classification block
		if pretrained:
			for layer in base_model.layers:
				layer.trainable=False
		x = GlobalAveragePooling2D(name='avg_pool')(base_model.output)
		x = Dense(output_dim, activation='softmax', name='predictions')(x)
		model = Model(inputs=base_model.input, outputs=x)
	elif architecture == "DenseNet":
		base_model = DenseNet201(weights=weights, include_top=False, input_shape = (input_dim))
		if pretrained:
			for layer in base_model.layers:
				layer.trainable=False
		x = GlobalAveragePooling2D(name='avg_pool')(base_model.output)
		x = Dense(output_dim, activation='softmax', name='predictions')(x)
		model = Model(inputs=base_model.input, outputs=x)
	model.summary()
	return(model)

def compile_and_save(model, train_generator, validation_generator, optimizer, learning_rate, loss, epochs, model_name):
	print(model, train_generator, validation_generator, optimizer, learning_rate, loss, epochs, model_name)

	model.summary()

	if optimizer == 'SGD':
		optimizer = SGD(lr=learning_rate)
	if optimizer == 'Adam':
		optimizer = Adam(lr=learning_rate)
	if optimizer == 'Adagrad':
		optimizer = Adagrad(lr=learning_rate)
	if optimizer == 'RMSprop':
		optimizer = RMSprop(lr=learning_rate)
	if loss == 'categorical_crossentropy':
		model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
	if loss == 'Dice':
		model.compile(optimizer=optimizer, loss=dice_coef_loss, metrics=[dice_coef])
	print('Model compiled')
	# exit()
	history = model.fit_generator(generator=train_generator,
	        epochs=epochs,
	        validation_data=validation_generator, callbacks=[LossHistory()])

	if not os.path.exists('../bin/' + model_name):
		os.mkdir('../bin/' + model_name)


	f = open('../bin/' + model_name + '/store.pckl', 'wb')
	pickle.dump(history, f)
	f.close()

	model_json = model.to_json()
	with open("../bin/" + model_name + "/model.json", "w") as json_file:
	    json_file.write(model_json)
	# serialize weights to HDF5
	model.save_weights("../bin/" + model_name + "/weights.h5")
	print("Saved model to disk")

def train_model(train_params, val_params, model_params, compile_params):
	train_generator, validation_generator = batch_generator(train_params, val_params)
	model = get_model(**model_params)
	compile_and_save(model, train_generator, validation_generator, **compile_params)


# Parameters
# architecture = 'VGG'
# width, height = 100, 100
# color_mode = 3
# num_classes = 5
# batch_size = 32
# params = {'data_path' : '/media/santis/5800-7A46/Histo_img_notordered',
# 		'dim': (width,height),
# 		'batch_size': batch_size,
# 		'n_classes': num_classes,
# 		'n_channels': 3,
# 		'shuffle': True,
# 		'data_aug': True}
          
# val_params = {'data_path' : '/media/santis/5800-7A46/Histo_img_notordered',
# 		'dim': (width,height),
# 		'batch_size': batch_size,
# 		'n_classes': num_classes,
# 		'n_channels': 3,
# 		'shuffle': True,
# 		'data_aug': False}

# net_params ={'architecture' : architecture,
# 			'input_dim' : (width, height, color_mode),
# 			'output_dim' : num_classes,
# 			'pre_trained' : False}

# compile_params = {'optimizer' : 'SGD',
# 				'learning_rate' : 0.0001,
# 				'loss' : 'categorical_crossentropy',
# 				'epochs' : 20	,
# 				'model_name' : 'densenet_v2'}



