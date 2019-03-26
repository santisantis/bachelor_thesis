# -*- coding: utf-8 -*-
import sys
from PyQt4 import QtGui, uic
from load import load_image, predict
# from train import batch_generator
from train_new import train_model
from train_new import *
from PyQt4.QtCore import QObject, pyqtSignal, pyqtSlot
 

class MyWindow(QtGui.QTabWidget):

	def __init__(self):
		super(MyWindow, self).__init__()
		uic.loadUi('../bin/gui/untitled.ui', self)
		self.train_params = {}
		self.val_params = {}

		self.browse_train_bt.clicked.connect(self.browse_train)
		self.browse_val_bt.clicked.connect(self.browse_val)
		self.size_combo.activated.connect(self.get_size)
		self.color_combo.activated.connect(self.get_color)
		self.batch_spin.valueChanged.connect(self.get_batch_size)
		self.aug_check.stateChanged.connect(self.get_aug)
		self.shuffle_data_check.stateChanged.connect(self.get_shuffle)
		self.pre_trained_check.stateChanged.connect(self.get_pretrained)

		self.model_params = {}
		self.compile_params = {}
		self.lr_input.textChanged.connect(self.get_lr)
		self.model_name.textChanged.connect(self.get_model_name)
		self.epoch_spin.valueChanged.connect(self.get_epoch)
		self.class_spin.valueChanged.connect(self.get_numclass)
		self.loss_func_combo.activated.connect(self.get_loss)
		self.optimizer_combo.activated.connect(self.get_opt)
		self.arch_combo.activated.connect(self.get_arch)
		self.train_bt.clicked.connect(self.confirm_train)

		self.tab3 = {}
		self.browse_model_bt.clicked.connect(self.browse_model)
		self.browse_weights_bt.clicked.connect(self.browse_weights)
		self.browse_photo_bt.clicked.connect(self.browse_photo)
		self.results_bt.clicked.connect(self.results)

		self.show()



	def get_size(self):
		if str(self.size_combo.currentText()) == '150x150':
			self.train_params['dim'] = (150,150)
			self.val_params['dim'] = (150,150)
		if str(self.size_combo.currentText()) == '200x200':
			self.train_params['dim'] = (200,200)
			self.val_params['dim'] = (200,200)
		if str(self.size_combo.currentText()) == '250x250':
			self.train_params['dim'] = (250,250)
			self.val_params['dim'] = (250,250)
		if str(self.size_combo.currentText()) == '300x300':
			self.train_params['dim'] = (300,300)
			self.val_params['dim'] = (300,300)
		print(self.train_params)

	def get_color(self):
		if str(self.color_combo.currentText()) == 'BW':
			self.train_params['n_channels'] = 1
			self.val_params['n_channels'] = 1
			self.model_params['input_dim'] = self.model_params['input_dim'] + (1,)
		if str(self.color_combo.currentText()) == 'RGB':
			self.train_params['n_channels'] = 3
			self.val_params['n_channels'] = 3
			self.model_params['input_dim'] = self.model_params['input_dim'] + (3,)
		if str(self.color_combo.currentText()) == 'RGBA':
			self.train_params['n_channels'] = 4
			self.val_params['n_channels'] = 4
			self.model_params['input_dim'] = self.model_params['input_dim'] + (4,)
		print(self.train_params)


	def get_batch_size(self, value):
		self.train_params['batch_size'] = value
		self.val_params['batch_size'] = value
		# print(self.train_params)
	def get_aug(self, value):
		self.train_params['data_aug'] = bool(value)
		self.val_params['data_aug'] = False
		# print(self.train_params)

	def get_shuffle(self, value):
		self.train_params['shuffle'] = bool(value)
		self.val_params['shuffle'] = bool(value)
		# print(self.train_params)

	def browse_train(self, *args, **kwargs):
		dlg = QtGui.QFileDialog(self)
		dlg.setFileMode(QtGui.QFileDialog.Directory)
		self.train_params['data_path'] = str(dlg.getExistingDirectory(dlg, "Wybierz folder", '/home/santis/Documents'))
		self.train_path_disp.setText(self.train_params['data_path'])
		# print(self.train_params)

	def browse_val(self, *args, **kwargs):
		dlg = QtGui.QFileDialog(self)
		dlg.setFileMode(QtGui.QFileDialog.Directory)
		self.val_params['data_path'] = str(dlg.getExistingDirectory(dlg, "Wybierz folder", '/home/santis/Documents'))
		self.val_path_disp.setText(self.val_params['data_path'])
		# print(self.val_params)

	def get_arch(self):
		self.model_params['architecture'] = str(self.arch_combo.currentText())	

	def get_pretrained(self, value):
		self.model_params['pretrained'] = bool(value)

	def get_numclass(self, value):
		self.train_params['n_classes'] = value
		self.val_params['n_classes'] = value
		self.model_params['output_dim'] = value


	def get_model_name(self, value):
		self.compile_params['model_name'] = str(value)


	def get_lr(self, value):
		self.compile_params['learning_rate'] = float(value)


	def get_epoch(self, value):
		self.compile_params['epochs'] = value


	def get_loss(self):
		self.compile_params['loss'] = str(self.loss_func_combo.currentText())

	def get_opt(self):
		self.compile_params['optimizer'] = str(self.optimizer_combo.currentText())


	def confirm_train(self):
		if not 'dim' in self.train_params:
			self.train_params['dim'] = (200,200)
			self.val_params['dim'] = (200,200)
		if not 'n_classes' in self.train_params:
			self.train_params['n_classes'] = self.class_spin.value()
			self.val_params['n_classes'] = self.class_spin.value()
		if not 'n_channels' in self.train_params:
			self.train_params['n_channels'] = 3
			self.val_params['n_channels'] = 3
		if not 'shuffle' in self.train_params:
			self.train_params['shuffle'] = True
			self.val_params['shuffle'] = True			
		if not 'data_aug' in self.train_params:
			self.train_params['data_aug'] = True
			self.val_params['data_aug'] = False
		if not 'batch_size' in self.train_params:
			self.train_params['batch_size'] = 32
			self.val_params['batch_size'] = 32
		if not 'data_path' in self.train_params:
			print("Error: Podaj gdzie znajduja sie obrazy treningowe")
		if not 'data_path' in self.val_params:
			print("Error: Podaj gdzie znajduja sie obrazy walidacyjne")
		print(self.train_params, self.val_params)


		if not 'input_dim' in self.model_params:
			self.model_params['input_dim'] = (200,200,3)
		if not 'architecture' in self.model_params:
			self.model_params['architecture'] = str(self.arch_combo.currentText())
		if not 'output_dim' in self.model_params:	
			self.model_params['output_dim'] = self.class_spin.value()
		if not 'pretrained' in self.model_params:	
			self.model_params['pretrained'] = False
		print(self.model_params)

		if not 'learning_rate' in self.compile_params:
			self.compile_params['learning_rate'] = 0.0001
		if not 'epochs' in self.compile_params:
			self.compile_params['epochs'] = self.epoch_spin.value()
		if not 'loss' in self.compile_params:
			self.compile_params['loss'] = str(self.loss_func_combo.currentText())
		if not 'optimizer' in self.compile_params:
			self.compile_params['optimizer'] = str(self.optimizer_combo.currentText())
		if not 'model_name' in self.compile_params:
			self.compile_params['model_name'] = 'Model1'
		print(self.compile_params)

		self.train_results_disp.setText('Trening trwa...')
		train_model(self.train_params, self.val_params, self.model_params, self.compile_params)
		self.train_results_disp.setPixmap(QtGui.QPixmap('/home/santis/Documents/Praca_inzynierska/bin/gui/train_results.png'))

	def browse_model(self, *args, **kwargs):
		dlg = QtGui.QFileDialog(self)
		dlg.setFileMode(QtGui.QFileDialog.AnyFile)

		self.tab3['model_path'] = str(dlg.getOpenFileName(dlg, "Wybierz model", '/home/santis/Documents', "Model (*.json)"))
		print(self.tab3)

	def browse_weights(self, *args, **kwargs):
		dlg = QtGui.QFileDialog(self)
		dlg.setFileMode(QtGui.QFileDialog.AnyFile)
		dlg.setFilter("Wagi modelu (*.h5)")
		self.tab3['weights_path']= str(dlg.getOpenFileName(dlg, "Wybierz wagi modelu", '/home/santis/Documents', "Wagi modelu (*.h5)"))
		print(self.tab3)

	def browse_photo(self, *args, **kwargs):
		dlg = QtGui.QFileDialog(self)
		dlg.setFileMode(QtGui.QFileDialog.AnyFile)
		self.tab3['img_path'] = str(dlg.getOpenFileName(dlg, "Wybierz obraz", '/home/santis/Pictures'))

		self.tab3['img_path'] = load_image(self.tab3['img_path'])
		self.photo_disp.setPixmap(QtGui.QPixmap(self.tab3['img_path']))
		# self.photo_disp.resize(100,100)
		self.photo_disp.show()
		# print(self.tab3)
	def results(self):
		self.results_disp.setText(u'Proszę czekać...')
		self.results_disp.setPixmap(QtGui.QPixmap(predict(self.tab3)))
		self.results_disp.show()
		
if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    window = MyWindow()

    sys.exit(app.exec_())