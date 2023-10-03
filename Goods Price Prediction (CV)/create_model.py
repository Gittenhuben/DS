import numpy as np
import random
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow import random as tf_random


SIZE_X = 192
SIZE_Y = 320


SEED = 100

random.seed(SEED)
np.random.seed(SEED)
tf_random.set_seed(SEED)



def create_model():

    backbone = ResNet50(input_shape=(SIZE_Y,SIZE_X,3),
                        weights='imagenet',
                        include_top=False) 

    model = Sequential()
    model.add(backbone)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(1, activation='relu')) 

    optimizer=Adam(learning_rate=0.0001)

    model.compile(optimizer=optimizer, loss='mean_squared_error',
                  metrics=['mse'])
    return model


model0 = create_model()

model0.save('model0.keras')
