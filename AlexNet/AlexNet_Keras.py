# importing libraries
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras import backend as K

class AlexNet:
	@staticmethod
	def build(width, height, depth, classes):

	# initialize the model along with input shape to be "channels last" and the 
	# channels dimensions itself
	model = Sequential()
	inputShape = (height, width, depth)
	chanDim = -1

	if K.image_data_format() == "channels_first":
		inputShape = (depth, height, width)
		chanDim = 1
	
	# Block #1: first CONV => RELU => POOL layer set
	model.add(Conv2D(96, (11, 11), strides = (4, 4), input_shape = inputShape, 
		padding = "valid"))
	model.add(Activation("relu"))
	model.add(BatchNormalization(axis = chanDim))
	model.add(MaxPooling2D(pool_size = (3, 3), strides = (2, 2)))

	# Block #2: second CONV => RELU => POOL layer set
	model.add(Conv2D(256, (5, 5), strides = (1, 1), padding="same"))
	model.add(Activation("relu"))
	model.add(BatchNormalization(axis = chanDim))
	model.add(MaxPooling2D(pool_size = (3, 3), strides = (2, 2)))

	# Block #3: CONV => RELU => CONV => RELU => CONV => RELU
	model.add(Conv2D(384, (3, 3), strides = (1, 1), padding = "same"))
	model.add(Activation("relu"))
	model.add(BatchNormalization(axis = chanDim))

	model.add(Conv2D(384, (3, 3), strides = (1, 1), padding="same"))
	model.add(Activation("relu"))
	model.add(BatchNormalization(axis = chanDim))

	model.add(Conv2D(256, (3, 3), strides = (1, 1), padding = "same"))
	model.add(Activation("relu"))
	model.add(BatchNormalization(axis=chanDim))

	model.add(MaxPooling2D(pool_size = (3, 3), strides = (2, 2)))
	model.add(Dropout(0.5))

	# Block #4: first set of FC => RELU layers
	model.add(Flatten())
	model.add(Dense(4096))
	model.add(Activation("relu"))
	model.add(BatchNormalization())
	model.add(Dropout(0.5))

	# Block #5: second set of FC => RELU layers
	model.add(Dense(4096))
	model.add(Activation("relu"))
	model.add(BatchNormalization())

	# softmax classifier
	model.add(Dense(classes))
	model.add(Activation("softmax"))

	# return the constructed network architecture
	return model