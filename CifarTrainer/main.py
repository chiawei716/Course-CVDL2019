import sys
import os, random
import tensorflow as tf
import tf.keras as keras
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np
cifar10 = tf.keras.datasets.cifar10

from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5 import QtWidgets, uic
from tensorboard.plugins.hparams import api as hp


label_dict = {
	0 : 'airplane'	,
	1 : 'automobile',
	2 : 'bird'      ,
	3 : 'cat'       ,
	4 : 'deer'      ,
	5 : 'dog'       ,
	6 : 'frog'      ,
	7 : 'horse'     ,
	8 : 'ship'      ,
	9 : 'truck'
}


# Load Cifar-10 datasets from keras
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Shuffle training datasets
train_num = len(x_train)
train_index = np.arange(train_num)
np.random.shuffle(train_index)
x_train = x_train[train_index]
y_train = y_train[train_index]

# Shuffle testing datasets
test_num = len(x_test)
test_index = np.arange(test_num)
np.random.shuffle(test_index)
x_test = x_test[test_index]
y_test = y_test[test_index]

print(len(x_train[0][0]))


# Load ui
path = os.getcwd()
qtCreatorFile = path + os.sep + "mainwindow.ui"
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile) 

# Build model
def conv2d(x, ft, b, strides, padding):
    x = tf.nn.conv2d(x, ft, strides=[1, strides, strides, 1], padding=padding)
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k, padding):
    return tf.nn.max_pool(
        x,
        ksize=[1, k, k, 1],
        strides=[1, k, k, 1],
        padding=padding)

# Set up ui
class MainWindow(QMainWindow, Ui_MainWindow):
	def __init__(self, parent=None):
		super(MainWindow, self).__init__(parent)
		self.setupUi(self)
		self.onBindingUI()
		
		self.batch_size = 32
		self.learning_rate = 0.001
		self.optimizer = 'sgd'
		
		self.HP_BATCH_SIZE = hp.HParam('num_units', hp.Discrete([self.batch_size, self.batch_size]))
		self.HP_LEARNING_RATE = hp.HParam('dropout', hp.RealInterval(self.learning_rate, self.learning_rate))
		self.HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete([self.optimizer]))
	

	def onBindingUI(self):
		self.showImages_btn.clicked.connect(self.showImages)
		self.showHyperparameters_btn.clicked.connect(self.showHyperparameters)
		self.trainEpoch_btn.clicked.connect(self.trainEpoch)
	# Show ten pictures randomly from datasets
	def showImages(self):
		for i in range(0, 10):
			idx = random.randint(0, train_num)
			ax = plt.subplot(2, 5, i+1)
			ax.imshow(x_train[idx], cmap='binary')
			title = label_dict[(y_train[idx][0])]
			ax.set_title(title, fontsize=10, y=-0.3)
			ax.axis('off')
		plt.show()
		plt.draw()
	
	def showHyperparameters(self):
		print('hyperparameters:')
		print('batch size: ' + str(self.batch_size))
		print('learning rate: ' + str(self.learning_rate))
		print('optimizer: ' + str.upper(self.optimizer))
	
	def: trainEpoch(self):
		model = tf.keras.models.Sequential([
    		tf.keras.layers.Flatten(),
			tf.keras.layers.Dense(hparams[HP_NUM_UNITS], activation=tf.nn.relu),
    		tf.keras.layers.Dropout(hparams[HP_DROPOUT]),
    		tf.keras.layers.Dense(10, activation=tf.nn.softmax),
  		])
		
	
		

if __name__ == "__main__":
	app = QApplication(sys.argv)
	window = MainWindow()
	window.show()
	sys.exit(app.exec_())
