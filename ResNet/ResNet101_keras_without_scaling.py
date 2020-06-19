# importing libraries
from keras.layers import Input, Add, add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D
from keras.models import Model
import keras.backend as K

def identity_block(input_tensor, kernel_size, filters, stage, block):
    """
        The identity_block is the block that has no conv layer at shortcut
    
        Arguments:
            input_tensor -- input tensor
            kernel_size: defualt 3, the kernel size of middle conv layer at main path
            filters: list of integers, the nb_filters of 3 conv layer at main path
            stage: integer, current stage label, used for generating layer names
            block: 'a','b'..., current block label, used for generating layer names
    
        Returns:
            X -- output of the identity block
    """
    eps = 1.1e-5

    # Retrieve Filters
    filters1, filters2, filters3 = filters

    # defining name basis
    conv_name_base = 'res101' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    chanDim = 3
    if K.image_data_format() == "channels_first":
        inputShape = (depth, height, width)
        chanDim = 1
  
    # First component of main path
    X = Conv2D(filters = filters1, kernel_size = (1, 1), name = conv_name_base + '2a')(input_tensor)
    X = BatchNormalization(epsilon = eps, axis = chanDim, name = bn_name_base + '2a')(X)
    X = Activation('relu', name = conv_name_base + '2a_relu')(X)
  
    # Second component of main path 
    X = ZeroPadding2D(padding = (1, 1), name = conv_name_base + '2b_zeropadding')(X)
    X = Conv2D(filters = filters2, kernel_size = (kernel_size, kernel_size), name = conv_name_base + '2b')(X)
    X = BatchNormalization(epsilon = eps, axis = chanDim, name = bn_name_base + '2b')(X)
    X = Activation('relu', name = conv_name_base + '2b_relu')(X)
  
    # Third component of main path 
    X = Conv2D(filters = filters3, kernel_size = (1, 1), name = conv_name_base + '2c')(X)
    X = BatchNormalization(epsilon = eps, axis = chanDim, name = bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, input_tensor])
    X = Activation('relu', name = 'res101' + str(stage) + block + '_relu')(X)
  
    return X

def convolutional_block(input_tensor, kernel_size, filters, stage, block, strides = (2, 2)):
    """
        conv_block is the block that has a conv layer at shortcut
    
        Arguments:
            input_tensor: input tensor
            kernel_size: defualt 3, the kernel size of middle conv layer at main path
            filters: list of integers, the nb_filters of 3 conv layer at main path
            stage: integer, current stage label, used for generating layer names
            block: 'a','b'..., current block label, used for generating layer names
            strides: Integer, specifying the stride to be used
    
        Returns:
            X: output of the convolutional block
    """
    chanDim = 3
    eps = 1.1e-5
  
    if K.image_data_format() == "channels_first":
        inputShape = (depth, height, width)
        chanDim = 1

    # defining name basis
    conv_name_base = 'res101' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    filters1, filters2, filters3 = filters
  
    ##### MAIN PATH #####
    # First component of main path 
    X = Conv2D(filters = filters1, kernel_size = (1, 1), strides = strides, name = conv_name_base + '2a')(input_tensor)
    X = BatchNormalization(epsilon = eps, axis = chanDim, name = bn_name_base + '2a')(X)
    X = Activation('relu', name = conv_name_base + '2a_relu')(X)
  
    # Second component of main path 
    X = ZeroPadding2D(padding = (1, 1), name = conv_name_base + '2b_zeropadding')(X)
    X = Conv2D(filters = filters2, kernel_size = (kernel_size, kernel_size), name = conv_name_base + '2b')(X)
    X = BatchNormalization(epsilon = eps, axis = chanDim, name = bn_name_base + '2b')(X)
    X = Activation('relu', name = conv_name_base + "2b_relu")(X)
  
    # Third component of main path 
    X = Conv2D(filters = filters3, kernel_size = (1, 1), name = conv_name_base + '2c')(X)
    X = BatchNormalization(epsilon = eps, axis = chanDim, name = bn_name_base + '2c')(X)
  
    ##### SHORTCUT PATH #### 
    X_shortcut = Conv2D(filters = filters3, kernel_size = (1, 1), strides = strides, name = conv_name_base + '1')(input_tensor)
    X_shortcut = BatchNormalization(epsilon = eps, axis = chanDim, name = bn_name_base + '1')(X_shortcut)
  
    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu', name = "res101" + str(stage) + block + "_relu")(X) 
  
    return X

class ResNet101:
    @staticmethod
    def build(input_shape = (224, 224, 3), classes = 1000):
        """
            Implementation of the popular ResNet50 the following architecture:
            CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3 -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER
    
            Arguments:
                input_shape -- shape of the images of the dataset
                classes -- integer, number of classes

            Returns:
                model -- a Model() instance in Keras
        """
        eps = 1.1e-5
        chanDim = 3
        X_input = Input(shape = input_shape, name = "data")
    
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            X_input = Input(shape = input_shape, name = "data")
            chanDim = 1
    
        # Define the input as a tensor with shape input_shape
        X_input = Input(shape = input_shape)
    
        # Zero-Padding
        X = ZeroPadding2D(padding = (3, 3), name = 'conv1_zeropadding')(X_input)
    
        # Stage 1
        X = Conv2D(filters = 64, kernel_size = (7, 7), strides = (2, 2), name = 'conv1')(X)
        X = BatchNormalization(epsilon = eps, axis = chanDim, name = 'bn_conv1')(X)
        X = Activation('relu', name = "conv1_relu")(X)
        X = ZeroPadding2D(padding = (1, 1), name = 'pool1_pad')(X)
        X = MaxPooling2D(pool_size = (3, 3), strides = (2, 2), name = 'pool1')(X)
    
        # Stage 2
        X = convolutional_block(input_tensor = X, kernel_size = 3, filters = [64, 64, 256], stage = 2, block = 'a', strides = (1, 1))
        X = identity_block(input_tensor = X, kernel_size = 3, filters = [64, 64, 256], stage = 2, block = 'b')
        X = identity_block(input_tensor = X, kernel_size = 3, filters = [64, 64, 256], stage = 2, block = 'c')
    
        # Stage 3 
        X = convolutional_block(input_tensor = X, kernel_size = 3, filters = [128, 128, 512], stage = 3, block = 'a')
        for i in range(1, 4):
            X = identity_block(input_tensor = X, kernel_size = 3, filters = [128, 128, 512], stage = 3, block = 'b' + str(i))
    
        # Stage 4
        X = convolutional_block(input_tensor = X, kernel_size = 3, filters = [256, 256, 1024], stage = 4, block='a')
        for i in range(1, 23):
            X = identity_block(input_tensor = X, kernel_size = 3, filters = [256, 256, 1024], stage = 4, block = 'b' + str(i))
    
    
        # Stage 5
        X = convolutional_block(input_tensor = X, kernel_size = 3, filters = [512, 512, 2048], stage = 5, block = 'a')
        X = identity_block(input_tensor = X, kernel_size = 3, filters = [512, 512, 2048], stage = 5, block = 'b')
        X = identity_block(input_tensor = X, kernel_size = 3, filters = [512, 512, 2048], stage = 5, block = 'c')
    
        # AVGPOOL 
        X = AveragePooling2D(pool_size = (7, 7), name = 'avg_pool')(X)
    
        ### END CODE HERE ###
    
        # output layer
        X = Flatten()(X)
        X = Dense(classes, activation = 'softmax', name = 'fc' + str(classes))(X)
    
        # Create model
        model = Model(inputs = X_input, outputs = X)
        return model

model = ResNet101.build(input_shape = (224, 224, 3), classes = 1000)
print(model.summary())