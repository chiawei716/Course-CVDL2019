import sys
import os, random
import tensorflow as tf
import matplotlib.pyplot as plt
cifar10 = tf.keras.datasets.cifar10

from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5 import QtWidgets, uic

batch_size = 32

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
x_train, x_test = x_train/255.0, x_test/255.0

# Load ui
path = os.getcwd()
qtCreatorFile = path + os.sep + "mainwindow.ui"
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile) 

# Set up ui
class MainWindow(QMainWindow, Ui_MainWindow):
	def __init__(self, parent=None):
		super(MainWindow, self).__init__(parent)
		self.setupUi(self)
		self.onBindingUI()


	def onBindingUI(self):
		self.showImage.clicked.connect(self.showImages)

	# Show ten pictures randomly from datasets
	def showImages(self):
		for i in range(0, 10):
			idx = random.randint(0,len(x_train)) 
			ax = plt.subplot(2, 5, i+1)
			ax.imshow(x_train[idx], cmap='binary')
			title = label_dict[(y_train[idx][0])]
			ax.set_title(title, fontsize=10, y=-0.3)
			ax.axis('off')
		plt.show()
		plt.draw()

if __name__ == "__main__":
	app = QApplication(sys.argv)
	window = MainWindow()
	window.show()
	sys.exit(app.exec_())
