# importing libraries
from keras.layers import Input, Add, add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D
from keras.models import Model
from keras.layers.core import Layer
from keras.engine import InputSpec
from keras import backend as K
try:
    from keras import initializations
except ImportError:
    from keras import initializers as initializations

class Scale(Layer):
    """
        Learns a set of weights and biases used for scaling the input data. the output consists simply in an 
        element-wise multiplication of the input and a sum of a set of constants:
        out = in * gamma + beta,
        where 'gamma' and 'beta' are the weights and biases larned.
        # Arguments
            axis: integer, axis along which to normalize in mode 0. For instance, if your input tensor has 
                  shape (samples, channels, rows, cols), set axis to 1 to normalize per feature map 
                  (channels axis).
            momentum: momentum in the computation of the exponential average of the mean and standard deviation 
                      of the data, for feature-wise normalization.
            weights: Initialization weights. List of 2 Numpy arrays, with shapes:
                     `[(input_shape,), (input_shape,)]`
            beta_init: name of initialization function for shift parameter 
                       (see [initializations](../initializations.md)), or alternatively,
                       Theano/TensorFlow function to use for weights initialization. 
                       This parameter is only relevant if you don't pass a `weights` argument.
            gamma_init: name of initialization function for scale parameter 
                        (see [initializations](../initializations.md)), or alternatively,
                        Theano/TensorFlow function to use for weights initialization.
                        This parameter is only relevant if you don't pass a `weights` argument.
    """
    def __init__(self, weights = None, axis = -1, momentum = 0.9, beta_init = 'zero', gamma_init = 'one', 
        **kwargs):
        self.momentum = momentum
        self.axis = axis
        self.beta_init = initializations.get(beta_init)
        self.gamma_init = initializations.get(gamma_init)
        self.initial_weights = weights
        super(Scale, self).__init__(**kwargs)
  
    def build(self, input_shape):
        self.input_spec = [InputSpec(shape = input_shape)]
        shape = (int(input_shape[self.axis]),)
    
        # Compatibility with TensorFlow >= 1.0.0
        self.gamma = K.variable(self.gamma_init(shape), name='{}_gamma'.format(self.name))
        self.beta = K.variable(self.beta_init(shape), name='{}_beta'.format(self.name))
        self.trainable_weights = [self.gamma, self.beta]
    
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
  
    def call(self, x, mask = None):
        input_shape = self.input_spec[0].shape
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis]
    
        out = K.reshape(self.gamma, broadcast_shape) * x + K.reshape(self.beta, broadcast_shape)
        return out
  
    def get_config(self):
        config = {"momentum": self.momentum, "axis": self.axis}
        base_config = super(Scale, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

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
    scale_name_base = 'scale' + str(stage) + block + '_branch'

    chanDim = 3
    if K.image_data_format() == "channels_first":
        inputShape = (depth, height, width)
        chanDim = 1
  
    # First component of main path
    X = Conv2D(filters = filters1, kernel_size = (1, 1), name = conv_name_base + '2a')(input_tensor)
    X = BatchNormalization(epsilon = eps, axis = chanDim, name = bn_name_base + '2a')(X)
    X = Scale(axis = chanDim, name = scale_name_base + '2a')(X)
    X = Activation('relu', name = conv_name_base + '2a_relu')(X)
  
    # Second component of main path 
    X = ZeroPadding2D(padding = (1, 1), name = conv_name_base + '2b_zeropadding')(X)
    X = Conv2D(filters = filters2, kernel_size = (kernel_size, kernel_size), name = conv_name_base + '2b')(X)
    X = BatchNormalization(epsilon = eps, axis = chanDim, name = bn_name_base + '2b')(X)
    X = Scale(axis = chanDim, name = scale_name_base + '2b')(X)
    X = Activation('relu', name = conv_name_base + '2b_relu')(X)
  
    # Third component of main path 
    X = Conv2D(filters = filters3, kernel_size = (1, 1), name = conv_name_base + '2c')(X)
    X = BatchNormalization(epsilon = eps, axis = chanDim, name = bn_name_base + '2c')(X)
    X = Scale(axis = chanDim, name = scale_name_base + '2c')(X)

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
    scale_name_base = 'scale' + str(stage) + block + '_branch'

    # Retrieve Filters
    filters1, filters2, filters3 = filters
  
    ##### MAIN PATH #####
    # First component of main path 
    X = Conv2D(filters = filters1, kernel_size = (1, 1), strides = strides, name = conv_name_base + '2a')(input_tensor)
    X = BatchNormalization(epsilon = eps, axis = chanDim, name = bn_name_base + '2a')(X)
    X = Scale(axis = chanDim, name = scale_name_base + '2a')(X)
    X = Activation('relu', name = conv_name_base + '2a_relu')(X)
  
    # Second component of main path 
    X = ZeroPadding2D(padding = (1, 1), name = conv_name_base + '2b_zeropadding')(X)
    X = Conv2D(filters = filters2, kernel_size = (kernel_size, kernel_size), name = conv_name_base + '2b')(X)
    X = BatchNormalization(epsilon = eps, axis = chanDim, name = bn_name_base + '2b')(X)
    X = Scale(axis = chanDim, name = scale_name_base + '2b')(X)
    X = Activation('relu', name = conv_name_base + "2b_relu")(X)
  
    # Third component of main path 
    X = Conv2D(filters = filters3, kernel_size = (1, 1), name = conv_name_base + '2c')(X)
    X = BatchNormalization(epsilon = eps, axis = chanDim, name = bn_name_base + '2c')(X)
    X = Scale(axis = chanDim, name = scale_name_base + '2c')(X)
  
    ##### SHORTCUT PATH #### 
    X_shortcut = Conv2D(filters = filters3, kernel_size = (1, 1), strides = strides, name = conv_name_base + '1')(input_tensor)
    X_shortcut = BatchNormalization(epsilon = eps, axis = chanDim, name = bn_name_base + '1')(X_shortcut)
    X_shortcut = Scale(axis = chanDim, name = scale_name_base + '1')(X_shortcut)

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
        X = Scale(axis = chanDim, name = 'scale_conv1')(X)
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