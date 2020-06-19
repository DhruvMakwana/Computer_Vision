# importing libraries
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras import backend as K


class VGG19:
    
    
    @staticmethod
    def build(width, height, depth, classes):

        # initialize the model along with input shape to be "channels last" and the 
        # channels dimensions itself
        model = Sequential()
        input_shape = (height, width, depth)
        if K.image_data_format() == "channels_first":
            input_shape = (depth, height, width)

        # Block 1:  CONV => RELU => CONV => RELU => POOL layer set
        model.add(Conv2D(64, (3, 3), input_shape=input_shape, padding='same'))
        model.add(Activation("relu")) 
        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size = (2,2), strides = (2, 2)))

        # Block 2: CONV => RELU => CONV => RELU => POOL layer set

        model.add(Conv2D(128, (3, 3), padding='same'))
        model.add(Activation("relu")) 
        model.add(Conv2D(128, (3, 3), padding='same'))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size = (2,2), strides = (2, 2)))

        # Block 3: CONV => RELU => CONV => RELU => CONV => RELU => POOL layer set

        model.add(Conv2D(256, (3, 3), padding='same'))
        model.add(Activation("relu")) 
        model.add(Conv2D(256, (3, 3), padding='same'))
        model.add(Activation("relu"))
        model.add(Conv2D(256, (3, 3), padding='same'))
        model.add(Activation("relu")) 
        model.add(Conv2D(256, (3, 3), padding='same'))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size = (2,2), strides = (2, 2)))

        # Block 4: CONV => RELU => CONV => RELU => CONV => RELU => POOL layer set
        model.add(Conv2D(512, (3, 3), padding='same'))
        model.add(Activation("relu")) 
        model.add(Conv2D(512, (3, 3), padding='same'))
        model.add(Activation("relu"))
        model.add(Conv2D(512, (3, 3), padding='same'))
        model.add(Activation("relu")) 
        model.add(Conv2D(512, (3, 3), padding='same'))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size = (2,2), strides = (2, 2)))

        # Block 5: CONV => RELU => CONV => RELU => CONV => RELU => POOL layer set

        model.add(Conv2D(512, (3, 3), padding='same'))
        model.add(Activation("relu")) 
        model.add(Conv2D(512, (3, 3), padding='same'))
        model.add(Activation("relu"))
        model.add(Conv2D(512, (3, 3), padding='same'))
        model.add(Activation("relu"))
        model.add(Conv2D(512, (3, 3), padding='same'))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size = (2,2), strides = (2, 2)))

        # Block 6: first set of FC => RELU layers

        model.add(Flatten())
        model.add(Dense(4096))
        model.add(Activation("relu"))
        model.add(Dropout(0.5))

        # Block 7: second set of FC => RELU layers

        model.add(Dense(4096))
        model.add(Activation("relu"))
        model.add(Dropout(0.5))

        # Softmax classifier

        model.add(Dense(classes))
        model.add(Activation("softmax"))
        
        return model