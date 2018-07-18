#!/bin/env python3
#SBATCH -N 1 # No. of computers you wanna use. Typically 1
#SBATCH -n 2 # No. of CPU cores you wanna use. Typically 1
#SBATCH -p gpu # This flag specifies that you wanna use GPU and not CPU
#SBATCH -o roadseg.out # output file name, in case your program has anything to output (like print, etc)
#SBATCH -t 24:00:00 # Amount of time
#SBATCH --gres=gpu:2 # No. of GPU cores you wanna use. Usually 2-3
import numpy
import pandas
import cv2
import math
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.constraints import maxnorm
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras import backend as K
from scipy.signal import savgol_filter
K.set_image_dim_ordering('th')

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

K = []
# load the dataset
dataframe = pandas.read_csv('road_dataset_newest.csv', header=1)
dataset = dataframe.values
data = dataset[10000:50000,:]

for i in data[:,0]:
	img = cv2.imread(i)
	img = cv2.resize(img, (36,36))
	K.append(img)
X = numpy.array(K)

#MA Filter
#N = 500
look_back = 60
Y = numpy.empty_like(data[:,2:4])
#imu_p_smooth_ma = dataset[25000-N:55000+N,2]
#imu_p_smooth_t = numpy.convolve(imu_p_smooth_ma, numpy.ones((N,))/N, mode='valid')
#imu_p_smooth_t = imu_p_smooth_t[int(N/2):int(len(imu_p_smooth_t)-(N/2))-1]

#SG filter
size = 91
imu_p_smooth = savgol_filter(dataset[10000:50000,2],size,2)

Y[:,0] = imu_p_smooth
Y[:,1] = data[:,3]

# split into train and test sets
train_size = int(len(data) * 0.67)
test_size = len(data) - train_size
X_train = X[0:(train_size)-look_back-1]
X_test = X[(train_size):len(X)-look_back-1]
train = data[0:train_size]
test = data[train_size:len(data)]
Y_train = Y[0:train_size,:]
Y_test = Y[train_size:len(dataset),:]
# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataY = []
	for i in range(len(dataset)-look_back-1):
		dataY.append(dataset[i + look_back, :])
	return numpy.array(dataY)
# reshape into X=t and Y=t+1
Y_train = create_dataset(Y_train, look_back)
Y_test = create_dataset(Y_test, look_back)
Y_train = Y_train.astype('float32')
Y_test = Y_test.astype('float32')
# reshape to be [samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], 3, 36, 36).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 3, 36, 36).astype('float32')
#X_train = X_train.astype('float32')
#X_test = X_test.astype('float32')
# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255
num_classes = 2
def model():
	# create model
	model = Sequential()
	model.add(Conv2D(32,3, 3, border_mode='same', input_shape=(3, 36, 36), activation='relu'))
	model.add(Dropout(0.15))
	model.add(Conv2D(32,3, 3, activation='relu', border_mode='same'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(64,3, 3, activation='relu', border_mode='same'))
	model.add(Dropout(0.15))
	model.add(Conv2D(64,3, 3, activation='relu', border_mode='same'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(128,3, 3, activation='relu', border_mode='same'))
	model.add(Dropout(0.15))
	model.add(Conv2D(128,3, 3, activation='relu', border_mode='same'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Flatten())
	model.add(Dropout(0.15))
	model.add(Dense(1024, activation='relu', W_constraint=maxnorm(3)))
	model.add(Dropout(0.15))
	model.add(Dense(512, activation='relu', W_constraint=maxnorm(3)))
	model.add(Dropout(0.15))
	model.add(Dense(num_classes))
	model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
	return model
# build the model
model = model()
# Fit the model
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), nb_epoch=10, batch_size=150, verbose=2)

# make predictions
trainPredict = model.predict(X_train)
testPredict = model.predict(X_test)
# calculate root mean squared error
trainScore_pitch = math.sqrt(mean_squared_error(Y_train[:,0], trainPredict[:,0]))
print('Pitch: Train Score: %.2f RMSE' % (trainScore_pitch))
testScore_pitch = math.sqrt(mean_squared_error(Y_test[:,0], testPredict[:,0]))
print('Pitch: Test Score: %.2f RMSE' % (testScore_pitch))
# calculate root mean squared error
trainScore_altitude = math.sqrt(mean_squared_error(Y_train[:,1], trainPredict[:,1]))
print('Altitude: Train Score: %.2f RMSE' % (trainScore_altitude))
testScore_altitude = math.sqrt(mean_squared_error(Y_test[:,1], testPredict[:,1]))
print('Altitude: Test Score: %.2f RMSE' % (testScore_altitude))

# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(data[:,2:4])
trainPredictPlot[:, 0] = numpy.nan
trainPredictPlot[:, 1] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, 0] = trainPredict[:,0]
trainPredictPlot[look_back:len(trainPredict)+look_back, 1] = Y_train[:,0]
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(data[:,2:4])
testPredictPlot[:, 0] = numpy.nan
testPredictPlot[:, 1] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(data)-1, 0] = testPredict[:,0]
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(data)-1, 1] = Y_test[:,0]
# plot baseline and predictions
plt.plot(trainPredictPlot, label = 'Training')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
plt.plot(testPredictPlot, label = 'Testing')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
plt.xlabel("Time in micro sec")
plt.ylabel("Slope (IMU Pitch)")
# plt.title("Slope Result 1")
plt.show()

# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(data[:,2:4])
trainPredictPlot[:, 0] = numpy.nan
trainPredictPlot[:, 1] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, 0] = trainPredict[:,1]
trainPredictPlot[look_back:len(trainPredict)+look_back, 1] = Y_train[:,1]
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(data[:,2:4])
testPredictPlot[:, 0] = numpy.nan
testPredictPlot[:, 1] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(data)-1, 0] = testPredict[:,1]
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(data)-1, 1] = Y_test[:,1]
# plot baseline and predictions
plt.plot(trainPredictPlot, label = 'Training')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
plt.plot(testPredictPlot, label = 'Testing')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
plt.xlabel("Time in micro sec")
plt.ylabel("Slope (GPS Altitude)")
# plt.title("Slope Result 2")
plt.show()

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
	json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
#print("Saved model to disk")
