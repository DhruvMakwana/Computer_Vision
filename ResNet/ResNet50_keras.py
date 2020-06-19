# importing libraries
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D
from keras.layers.core import Activation, Flatten, Dense
from keras.models import Model
from keras.layers import Add, Input
from keras import backend as K

def identity_block(input_tensor, kernel_size, filters, stage, block):
	"""
		Implementation of the identity block

		Arguments:

			input_tensor -- input tensor 
			kernel_size -- default 3, specifying the shape of the middle CONV's window for the main path
			filters -- python list of integers, defining the number of filters in the CONV layers of the 
					   main path
			stage -- integer, used to name the layers, depending on their position in the network
			block -- string/character, used to name the layers, depending on their position in the network
    
		Returns:
			X -- output of the identity block
	"""
	chanDim = -1

	if K.image_data_format() == "channels_first":
		inputShape = (depth, height, width)
		chanDim = 1

	# defining name basis
	conv_name_base = 'res' + str(stage) + block + '_branch'
	bn_name_base = 'bn' + str(stage) + block + '_branch'

	# Retrieve Filters
	filters1, filters2, filters3 = filters

	# First component of main path
	X = Conv2D(filters = filters1, kernel_size = (1, 1), strides = (1, 1), padding = 'valid', 
		name = conv_name_base + '2a', kernel_initializer = 'he_normal')(input_tensor)
	X = BatchNormalization(axis = chanDim, name = bn_name_base + '2a')(X)
	X = Activation('relu')(X)

	# Second component of main path 
	X = Conv2D(filters = filters2, kernel_size = kernel_size, strides = (1, 1), padding = 'same', 
		name = conv_name_base + '2b', kernel_initializer = "he_normal")(X)
	X = BatchNormalization(axis = chanDim, name = bn_name_base + '2b')(X)
	X = Activation('relu')(X)

	# Third component of main path 
	X = Conv2D(filters = filters3, kernel_size = (1, 1), strides = (1, 1), padding = 'valid', 
		name = conv_name_base + '2c', kernel_initializer = "he_normal")(X)
	X = BatchNormalization(axis = chanDim, name = bn_name_base + '2c')(X)

	# Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
	X = Add()([X, input_tensor])
	X = Activation('relu')(X)

	return X

def convolutional_block(input_tensor, kernel_size, filters, stage, block, strides = (2, 2)):
	"""
		Implementation of the convolutional block

		Arguments:
			input_tensor -- input tensor
			kernel_size -- default 3, specifying the shape of the middle CONV's window for the main path
			filters -- python list of integers, defining the number of filters in the CONV layers of the 
					   main path
			stage -- integer, used to name the layers, depending on their position in the network
			block -- string/character, used to name the layers, depending on their position in the network
			strides -- Integer, specifying the stride to be used
		
		Returns:
			X -- output of the convolutional block
	"""
	chanDim = -1
	if K.image_data_format() == "channels_first":
		inputShape = (depth, height, width)
		chanDim = 1

	# defining name basis
	conv_name_base = 'res' + str(stage) + block + '_branch'
	bn_name_base = 'bn' + str(stage) + block + '_branch'

	# Retrieve Filters
	filters1, filters2, filters3 = filters

	##### MAIN PATH #####
	# First component of main path 
	X = Conv2D(filters = filters1, kernel_size = (1, 1), strides = strides, name = conv_name_base + '2a', 
		kernel_initializer = "he_normal")(input_tensor)
	X = BatchNormalization(axis = chanDim, name = bn_name_base + '2a')(X)
	X = Activation('relu')(X)

	# Second component of main path 
	X = Conv2D(filters = filters2, kernel_size = kernel_size, strides = (1, 1), padding = 'same', 
		name = conv_name_base + '2b', kernel_initializer = "he_normal")(X)
	X = BatchNormalization(axis = chanDim, name = bn_name_base + '2b')(X)
	X = Activation('relu')(X)

	# Third component of main path 
	X = Conv2D(filters = filters3, kernel_size = (1, 1), strides = (1, 1), padding = 'valid', 
		name = conv_name_base + '2c', kernel_initializer = "he_normal")(X)
	X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

	##### SHORTCUT PATH #### 
	X_shortcut = Conv2D(filters = filters3, kernel_size = (1, 1), strides = strides, padding = 'valid', 
		name = conv_name_base + '1', kernel_initializer = "he_normal")(input_tensor)
	X_shortcut = BatchNormalization(axis = chanDim, name = bn_name_base + '1')(X_shortcut)

	# Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
	X = Add()([X, X_shortcut])
	X = Activation('relu')(X)

	return X

class ResNet50:
	@staticmethod
	def build(input_shape = (224, 224, 3), classes = 1000):
		"""
			Implementation of the popular ResNet50 the following architecture:
			CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3 
			-> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER
			Arguments:
				input_shape -- shape of the images of the dataset
				classes -- integer, number of classes

			Returns:
				model -- a Model() instance in Keras
		"""
		chanDim = -1
		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)
			chanDim = 1

		# Define the input as a tensor with shape input_shape
		X_input = Input(shape = input_shape)

		# Zero-Padding
		X = ZeroPadding2D(padding = (3, 3), name = 'conv1_pad')(X_input)

		# Stage 1
		X = Conv2D(filters = 64, kernel_size = (7, 7), strides = (2, 2), padding = "valid", name = 'conv1', 
			kernel_initializer = "he_normal")(X)
		X = BatchNormalization(axis = chanDim, name = 'bn_conv1')(X)
		X = Activation('relu')(X)
		X = ZeroPadding2D(padding = (1, 1), name = 'pool1_pad')(X)
		X = MaxPooling2D(pool_size = (3, 3), strides = (2, 2))(X)
    
		# Stage 2
		X = convolutional_block(input_tensor = X, kernel_size = 3, filters = [64, 64, 256], stage = 2, 
			block = 'a', strides = (1, 1))
		X = identity_block(input_tensor = X, kernel_size = 3, filters = [64, 64, 256], stage = 2, block = 'b')
		X = identity_block(input_tensor = X, kernel_size = 3, filters = [64, 64, 256], stage = 2, block = 'c')
    
		# Stage 3 
		X = convolutional_block(input_tensor = X, kernel_size = 3, filters = [128, 128, 512], stage = 3, 
			block = 'a')
		X = identity_block(input_tensor = X, kernel_size = 3, filters = [128, 128, 512], stage = 3, block = 'b')
		X = identity_block(input_tensor = X, kernel_size = 3, filters = [128, 128, 512], stage = 3, block = 'c')
		X = identity_block(input_tensor = X, kernel_size = 3, filters = [128, 128, 512], stage = 3, block = 'd')

		# Stage 4
		X = convolutional_block(input_tensor = X, kernel_size = 3, filters = [256, 256, 1024], stage = 4, 
			block='a')
		X = identity_block(input_tensor = X, kernel_size = 3, filters = [256, 256, 1024], stage = 4, block = 'b')
		X = identity_block(input_tensor = X, kernel_size = 3, filters = [256, 256, 1024], stage = 4, block = 'c')
		X = identity_block(input_tensor = X, kernel_size = 3, filters = [256, 256, 1024], stage = 4, block = 'd')
		X = identity_block(input_tensor = X, kernel_size = 3, filters = [256, 256, 1024], stage = 4, block = 'e')
		X = identity_block(input_tensor = X, kernel_size = 3, filters = [256, 256, 1024], stage = 4, block = 'f')

		# Stage 5
		X = convolutional_block(input_tensor = X, kernel_size = 3, filters = [512, 512, 2048], stage = 5, 
			block = 'a')
		X = identity_block(input_tensor = X, kernel_size = 3, filters = [512, 512, 2048], stage = 5, block = 'b')
		X = identity_block(input_tensor = X, kernel_size = 3, filters = [512, 512, 2048], stage = 5, block = 'c')

		# AVGPOOL
		X = AveragePooling2D(pool_size = (2, 2))(X)

		### END CODE HERE ###
		# output layer
		X = Flatten()(X)
		X = Dense(classes, activation = 'softmax', name = 'fc' + str(classes), kernel_initializer = "he_normal")(X)

		# Create model
		model = Model(inputs = X_input, outputs = X, name = 'ResNet50')
		return model

model = ResNet50.build(input_shape = (224, 224, 3), classes = 1000)
print(model.summary())