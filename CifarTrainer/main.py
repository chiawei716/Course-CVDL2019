import sys
import os, random
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import numpy as np

print('version :' , torch.__version__)
print('cuda :' , torch.cuda.is_available())
print('cudnn :' , torch.backends.cudnn.enabled)
from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5 import QtWidgets, uic
from tensorboard.plugins.hparams import api as hp

EPOCH = 100
BATCH_SIZE = 32
LR = 0.001
DOWNLOAD_data = True

train_data = torchvision.datasets.CIFAR10(
    root = './data',
    train = True,
    transform=torchvision.transforms.ToTensor(), #改成torch可讀
    download=DOWNLOAD_data,
)

print(len(train_data.train_data))
print(len(train_data.train_labels))
for i in range(5):    
    img = np.asarray(train_data.train_data[i])
    plt.imshow(img, cmap='gray')
    plt.title('%i' % train_data.train_labels[i])
    plt.show()

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


# # Shuffle training datasets
# train_num = len(x_train)
# train_index = np.arange(train_num)
# np.random.shuffle(train_index)
# x_train = x_train[train_index]
# y_train = y_train[train_index]

# # Shuffle testing datasets
# test_num = len(x_test)
# test_index = np.arange(test_num)
# np.random.shuffle(test_index)
# x_test = x_test[test_index]
# y_test = y_test[test_index]


# Load ui
path = os.getcwd()
qtCreatorFile = path + os.sep + "mainwindow.ui"
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile) 

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(       
            nn.Conv2d(1,6,(5, 5),1),
			nn.ReLU(),
            nn.MaxPool2d((2, 2), 2),              
        )
        self.conv2 = nn.Sequential(       
            nn.Conv2d(6,16,(5, 5),1),
			nn.ReLU(),
            nn.MaxPool2d((2, 2), 2),
        )
		self.conv3 = nn.Sequential(       
            nn.Conv2d(16,120,(5, 5),1),
			nn.ReLU(),
        )
		self.fc = nn.Sequential(
			nn.Linear(120, 84),
			nn.ReLU(),
			nn.Linear(84, 10)
			nn.LogSoftmax(dim=-1)
		)  
        
    def forward(self, x):
        output = self.conv1(output)
        output = self.conv2(output)
		output = self.conv3(output)
        output = output.view(output.size(0), -1)	# flatening
        output = self.fc(output)
        return output

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
			idx = random.randint(0, len(train_data.train_data))
			img = np.asarray(train_data.train_data[idx])
			ax = plt.subplot(2, 5, i+1)
			ax.imshow(train_data.train_data[idx], cmap='binary')
			title = label_dict[(train_data.train_labels[idx][0])]
			ax.set_title(title, fontsize=10, y=-0.3)
			ax.axis('off')
		plt.show()
		plt.draw()
	
	def showHyperparameters(self):
		print('hyperparameters:')
		print('batch size: ' + str(BATCH_SIZE))
		print('learning rate: ' + str(LR))
		print('optimizer: SGD')
	
	def trainEpoch(self):
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
