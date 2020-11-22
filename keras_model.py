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
import tensorflow.keras.models
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Activation

########################################################################
# keras model
########################################################################
def get_model(inputDim):
    """
    define the keras model
    the model based on the simple dense auto encoder 
    (128*128*128*128*8*128*128*128*128)
    """
    inputLayer = Input(shape=(inputDim,))
   
    h = Dense(256)(inputLayer)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)
    
    h = Dense(256)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)

    h = Dense(256)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)

    h = Dense(256)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)

    h = Dense(256)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)

    h = Dense(128)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)
    
    h = Dense(16)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)
    
    h = Dense(128)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)
    
    h = Dense(256)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)
    
    h = Dense(256)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)
    
    h = Dense(256)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)

    h = Dense(256)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)

    h = Dense(256)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)

    h = Dense(inputDim)(h)

    return Model(inputs=inputLayer, outputs=h)
#########################################################################


def load_model(file_path):
    return tensorflow.keras.models.load_model(file_path)


if __name__ == "__main__":
    from tensorflow.keras.utils import plot_model

    model = get_model(64)
    plot_model(model, to_file='model.png')
