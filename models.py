import tensorflow as tf
import tensorflow.python.keras
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, BatchNormalization, GlobalAveragePooling2D, Reshape, LeakyReLU, Dropout, UpSampling2D, Conv2DTranspose
from abc import ABC, abstractmethod
import math


class Abstract_Model(ABC):
    
    @abstractmethod
    def get_model(self, input_shape, n_classes, use_imagenet):
        pass
    
    def preprocess_input(self, X):
        pass
    
    
    #util function to construct autoencoders architecture
    #stack conv layer with batch normalization
    def build_block(self, model, filters, conv_layer, padding = None, input_shape = None):
        if (input_shape is not None) and (padding is not None):
            model.add(conv_layer(filters, kernel_size=(4,4), strides=(2,2), padding=padding, input_shape=input_shape))
            model.add(BatchNormalization(momentum=0.8))
        elif (input_shape is not None) and (padding is None):
            model.add(conv_layer(filters, kernel_size=(4,4), input_shape=input_shape))
            model.add(BatchNormalization(momentum=0.8))
        elif (padding is None):
            model.add(conv_layer(filters, kernel_size=(4,4)))
            model.add(BatchNormalization(momentum=0.8))
        else:
            model.add(conv_layer(filters, kernel_size=(4,4), strides=(2,2), padding=padding))
            model.add(BatchNormalization(momentum=0.8))
    
    
class Simple_Model(Abstract_Model):
    
    def __init__(self):
        self.model = None

    def get_model(self, input_shape, n_classes, use_imagenet = False):

        self.model = Sequential()

        #block 1
        self.build_block(self.model, 32, Conv2D, padding='same', input_shape=input_shape) 
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), name='MAXPOOL_BLOCK1'))

        #block 2
        self.build_block(self.model, 32, Conv2D, padding='same')
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), name='MAXPOOL_BLOCK2'))

        #block 3
        self.build_block(self.model, 64, Conv2D, padding='same')
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), name='MAXPOOL_BLOCK3'))

        #fully connected
        self.model.add(Flatten(name='flatten'))  # this converts our 3D feature maps to 1D feature vectors
        self.model.add(Dense(64, activation='relu', name='fc1'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(n_classes, activation='sigmoid', name='predictions'))

        return self.model
    
    def preprocess_input(self, X):
        return X
    
    def __repr__(self):
        return 'simpleCNNmodel'
    

class Inception_model(Abstract_Model):
    
    def __init__(self):
        self.model = None
        
    def get_model(self, input_shape, n_classes, use_imagenet = False):
        # load pre-trained model graph, don't add final layer
        base_model = keras.applications.InceptionV3(include_top=False, input_shape=input_shape,
                                              weights='imagenet' if use_imagenet else None)
        # add global pooling just like in InceptionV3
        new_output = keras.layers.GlobalAveragePooling2D()(base_model.output)
        # add new dense layer for our labels
        new_output = keras.layers.Dense(n_classes, activation='sigmoid')(new_output)
        self.model = keras.engine.training.Model(base_model.inputs, new_output)
        return self.model
    
    def preprocess_input(self, X):
        return keras.applications.inception_v3.preprocess_input(X)
    
    def __repr__(self):
        return 'inception_v3'
    

class ResNet_model(Abstract_Model):
    
    def __init__(self):
        self.model = None
        
    def get_model(self, input_shape, n_classes, use_imagenet = False):
        
        base_model = keras.applications.ResNet50(include_top=False, input_shape=input_shape, 
                                            weights='imagenet' if use_imagenet else None)
    
        x = base_model.output
        x = Flatten(name='flatten')(x)
        
        #use sigmoid for binary classification - softmax has trouble
        prediction = Dense(n_classes, activation='sigmoid')(x)

        self.model = keras.engine.training.Model(base_model.inputs, prediction)

        return self.model
    
    def preprocess_input(self, X):
        #needed to work fine
        return keras.applications.resnet50.preprocess_input(X)
    
    def __repr__(self):
        return 'resnet50'
    
    
class Generator_model(Abstract_Model):
    
    def __init__(self):
        self.model = None
    
    
    def get_model(self, input_shape, n_classes = 1, use_imagenet = False):
            net = Sequential()
            dropout_prob = 0.4

            net.add(Dense(8*8*512, input_dim=input_shape))
            net.add(BatchNormalization(momentum=0.9))
            net.add(LeakyReLU())
            net.add(Reshape((8, 8, 512)))
            net.add(Dropout(dropout_prob))

            net.add(UpSampling2D())
            net.add(Conv2DTranspose(512, 5, padding='same'))
            net.add(BatchNormalization(momentum=0.9))
            net.add(LeakyReLU())

            net.add(UpSampling2D())
            net.add(Conv2DTranspose(256, 5, padding='same'))
            net.add(BatchNormalization(momentum=0.9))
            net.add(LeakyReLU())

            net.add(UpSampling2D())
            net.add(Conv2DTranspose(128, 5, padding='same'))
            net.add(BatchNormalization(momentum=0.9))
            net.add(LeakyReLU())

            net.add(UpSampling2D())
            net.add(Conv2DTranspose(64, 5, padding='same'))
            net.add(BatchNormalization(momentum=0.9))
            net.add(LeakyReLU())

            net.add(UpSampling2D())
            net.add(Conv2D(3, 5, padding='same'))
            net.add(Activation('sigmoid'))
            
            self.model = net

            return self.model      
    
    def preprocess_input(self, X):
        return X        
    
    def __repr__(self):
        return 'generator_model'
   

        
class Generator_model_complex(Generator_model):
    
    def __init__(self, filters, code_shape, leaky_alpha=0.2):
        self.model = None
        self.filters = filters
        self.code_shape = code_shape
        self.leaky_alpha = leaky_alpha
    
    
    #input shape is the image shape you obtained.
    # input shape works with 2**n x 2**n x 3
    def get_model(self, input_shape, n_classes = 1, use_imagenet = False):
        decoder = Sequential()
        number_layers = int(math.log2(input_shape[0])) -1 
        for i in range(number_layers):
            if i == 0:
                self.build_block(decoder, self.filters*(2**(number_layers-i-2)), Conv2DTranspose, input_shape=self.code_shape)
                #decoder.add(Activation('relu'))
                decoder.add(LeakyReLU(alpha=self.leaky_alpha))
            elif i == number_layers-1:
                self.build_block(decoder, 3, Conv2DTranspose, padding='same')
                decoder.add(Activation(activation='tanh'))
            else:
                self.build_block(decoder, self.filters*(2**(number_layers-i-2)), Conv2DTranspose, padding='same')
                #decoder.add(Activation('relu'))
                decoder.add(LeakyReLU(alpha=self.leaky_alpha))
        
        self.model = decoder

        return self.model

    def __repr__(self):
        return 'generator_artDCGAN'
    
    

class Discriminator_model(Abstract_Model):
    
    def __init__(self,filters, code_shape, include_top = True, leaky_alpha = 0.2):
        self.model = None
        self.filters = filters
        self.code_shape = code_shape
        self.include_top = include_top
        self.leaky_alpha = leaky_alpha
        
    
    # input shape works with 2**n x 2**n x 3
    def get_model(self, input_shape, n_classes = 1, use_imagenet = False):
        encoder = Sequential()
        number_layers = int(math.log2(input_shape[0])) -1
        for i in range(number_layers):
            if i == 0:
                self.build_block(encoder, self.filters*(2**i), Conv2D, padding='same', input_shape=input_shape)
                encoder.add(LeakyReLU(alpha=self.leaky_alpha))
            elif i == (number_layers-1):
                self.build_block(encoder, self.code_shape, Conv2D)
                encoder.add(LeakyReLU(alpha=self.leaky_alpha))
            else:
                self.build_block(encoder,self.filters*(2**i), Conv2D, padding='same')
                encoder.add(LeakyReLU(alpha=self.leaky_alpha))
        
        if self.include_top:
            encoder.add(Flatten())
            encoder.add(Dense(n_classes, activation='sigmoid', name='predictions'))
        
        self.model = encoder

        return self.model

    def preprocess_input(self, X):
        return X
    
    def __repr__(self):
        return 'discriminator_artDCGAN'
