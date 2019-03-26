import matplotlib.pyplot as plt  
import pickle
import keras.losses
from keras import backend as K
import numpy as np
import keras.metrics

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1.) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

keras.losses.dice_coef_loss = dice_coef_loss
keras.metrics.dice_coef = dice_coef
f = open('../bin/DenseNet/store.pckl', 'rb')
history = pickle.load(f)
f.close()


temp = []

data = {}

# for key in history.history.keys():
# 	mean_list = []
# 	for i, value in enumerate(history.history[key]):
# 		temp.append(value)
# 		if len(temp) == 2:
# 			mean_list.append(sum(temp)/len(temp))
# 			temp = []
# 		if len(mean_list) == len(history.history[key])/2:
# 			data[key] = mean_list
# 			print()

	# if i == 49:
	# 	mean_list.append((history.history['val_loss'][49]+history.history['val_loss'][48]) / 2)


# exit()

# plt.figure(1)  
   
#  # summarize history for accuracy  
   
# plt.subplot(211)  
# plt.plot(history.history['dice_coef'])  
# plt.plot(history.history['val_dice_coef'])  
# plt.title('model accuracy')  
# plt.ylabel('accuracy')  
# plt.xlabel('epoch')  
# plt.legend(['train', 'test'], loc='upper left')  

# # summarize history for loss  

# plt.subplot(212)  
# plt.plot(history.history['loss'])  
# plt.plot(history.history['val_loss'])  
# plt.title('model loss')  
# plt.ylabel('loss')  
# plt.xlabel('epoch')  
# plt.legend(['train', 'test'], loc='upper left')  
# plt.show()  

x_val = list(np.linspace(0,10,5))
plt.figure(1)  

 # summarize history for accuracy  
   
plt.subplot(211)  
plt.plot(history.history['acc'])  
plt.plot(history.history['val_acc'])  
plt.tick_params(axis='both', labelsize=30)
# plt.plot(x_val, data['acc'])  
# plt.plot(x_val, data['val_acc'])  
plt.title('Wykres dokladnosci', size=30)  
plt.ylabel('dokladnsc', size=30)  
# plt.xlabel('epoch')  
plt.legend(['trening', 'walidacja'], loc='upper left')  

# summarize history for loss  

plt.subplot(212)  
plt.plot(history.history['loss'])  
plt.plot(history.history['val_loss']) 
plt.tick_params(axis='both', labelsize=30)
# plt.plot(x_val, data['loss'])  
# plt.plot(x_val, data['val_loss'])  
plt.title('Wykres funkcji kosztu', size=30)  
plt.ylabel('wartosc f. kosztu', size=30)  
plt.xlabel('epoka', size=30)  
plt.legend(['trening', 'walidacja'], loc='upper left')  
plt.show()  

