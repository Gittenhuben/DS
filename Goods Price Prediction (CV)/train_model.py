import os
import numpy as np
import pandas as pd
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.models import load_model
from tensorflow import random as tf_random



SIZE_X = 192
SIZE_Y = 320
BATCH_SIZE = 12
NROWS = None

SEED = 100

random.seed(SEED)
np.random.seed(SEED)
tf_random.set_seed(SEED)



def get_model_number():
    i = 0
    while os.path.exists("model%s.keras" % i):
        i += 1
    return(i)



def load_files(mode):
    labels = pd.read_csv('table2.csv', sep=';', decimal=',', nrows=NROWS)

    datagen = ImageDataGenerator(validation_split=0.25,
                                 rescale=1./255)

    datagen_flow = datagen.flow_from_dataframe(
        labels,
        directory='d:\\rrr3FULLDL\\',
        target_size=(SIZE_Y, SIZE_X),
        x_col='file_name',
        y_col='eff',
        batch_size=BATCH_SIZE,
        class_mode='raw',
        subset=mode,
        seed=SEED)

    return datagen_flow



def train_model(model, train_datagen_flow, test_datagen_flow, batch_size=None,
                epochs=1,
                steps_per_epoch=None,
                validation_steps=None):

    History = model.fit(train_datagen_flow,
                        validation_data=test_datagen_flow,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        shuffle=True,
                        steps_per_epoch=steps_per_epoch,
                        validation_steps=validation_steps)

    val_loss = History.history['val_loss']
    trn_loss = History.history['loss']

    file_log_mse = open("log1.txt", "a")
    file_log_mse.write(str(get_model_number()) + " : " + str(val_loss[0]) + " : " + str(trn_loss[0]) + "\n")
    file_log_mse.close()

    return model


model_number = get_model_number()

model1 = load_model('model%s.keras' % (model_number - 1))


model1 = train_model(model1, load_files('training'), load_files('validation'))

model1.save('model%s.keras' % model_number)
