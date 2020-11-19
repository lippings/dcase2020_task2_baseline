"""
 @file   keras_model.py
 @brief  Script for keras model definition
 @author Toshiki Nakamura, Yuki Nikaido, and Yohei Kawaguchi (Hitachi Ltd.)
 Copyright (C) 2020 Hitachi, Ltd. All right reserved.
"""

########################################################################
# import python-library
########################################################################
# from import
import keras.models
from keras.models import Model
from keras.layers import Input, Dense, BatchNormalization, Activation

from gradient_reversal.flipGradientTF import GradientReversal

########################################################################
# keras model
########################################################################
def get_model(inputDim, daDim):
    """
    define the keras model
    the model based on the simple dense auto encoder 
    (128*128*128*128*8*128*128*128*128)
    """
    inputLayer = Input(shape=(inputDim,))
   
    h = Dense(256)(inputLayer)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)
   
    h = Dense(256)(inputLayer)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)
   
    h = Dense(256)(inputLayer)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)
    
    h = Dense(256)(inputLayer)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)

    h = Dense(256)(inputLayer)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)

    h = Dense(256)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)
    
    h = Dense(8)(h)
    h = BatchNormalization()(h)
    aux = Activation('relu')(h)

    h = Dense(256)(aux)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)

    h = Dense(256)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)
    
    h = Dense(256)(inputLayer)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)
    
    h = Dense(256)(inputLayer)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)
    
    h = Dense(256)(inputLayer)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)
    
    h = Dense(256)(inputLayer)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)

    h = Dense(inputDim, name='autoencoder')(h)

    da_in = GradientReversal(0.5)(aux)

    da_h = Dense(128)(da_in)
    da_h = BatchNormalization()(da_h)
    da_h = Activation('relu')(da_h)

    da_h = Dense(128)(da_h)
    da_h = BatchNormalization()(da_h)
    da_h = Activation('relu')(da_h)

    da_h = Dense(daDim, name='domain_classifier')

    return Model(inputs=inputLayer, outputs=[h, da_h])
#########################################################################


def load_model(file_path):
    return tensorflow.keras.models.load_model(file_path)


if __name__ == "__main__":
    from tensorflow.keras.utils import plot_model

    model = get_model(64)
    plot_model(model, to_file='model.png')
