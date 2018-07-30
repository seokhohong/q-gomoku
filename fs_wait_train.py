
# coding: utf-8

# In[1]:


import numpy as np
from keras.layers import Input, Convolution2D, Dense, Dropout, Flatten, concatenate, BatchNormalization
from keras.models import Model  # basic class for specifying and training a neural network
from keras import losses
import keras
from keras.callbacks import EarlyStopping

import tensorflow as tf

import os.path
import time


# In[2]:


SIZE = 7
WIN_CHAIN_LENGTH = 5
CHANNELS = 20
EPOCHS = 100
BATCH_SIZE = 100
GAME_BATCH = 500

VECTORS_NPZ = 'gomoku/models/waiting_vectors.npz'
VECTORS_COMPLETE = 'gomoku/models/waiting_vectors_complete'
P_MODEL = "gomoku/models/waiting_p.model"
Q_MODEL = "gomoku/models/waiting_q.model"
MODEL_COMPLETE = 'gomoku/models/waiting_models_complete'

PATIENCE = 3


# In[3]:


def wait_to_read(file_path):
    while not os.path.exists(file_path):
        time.sleep(1)

    if os.path.isfile(file_path):
        return
    else:
        raise ValueError("%s isn't a file!" % file_path)


# In[4]:


def train_model(npz):
    
    q_model = keras.models.load_model(Q_MODEL)
    p_model = keras.models.load_model(P_MODEL)
    
    print('Loaded Models')
    
    train_p_vectors = npz['train_p_vectors']
    train_q_vectors = npz['train_q_vectors']
    train_p = npz['train_p']
    train_q = npz['train_q']
    
    print('Num p', len(train_p_vectors))
    print('Num q', len(train_q_vectors))

    with(tf.device('/gpu:0')):
        if len(train_p_vectors) > 0:
            q_model.fit(x=train_q_vectors,
                        y=train_q,
                        shuffle=True,
                        callbacks=[EarlyStopping(patience=PATIENCE)],
                        validation_split = 0.1,
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS)
            # doesn't always need to train P
            p_model.fit(x=train_p_vectors,
                        y=train_p,
                        shuffle=True,
                        callbacks=[EarlyStopping(patience=PATIENCE)],
                        validation_split = 0.1,
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS)
    
    q_model.save(Q_MODEL)
    p_model.save(P_MODEL)
    
    with open(MODEL_COMPLETE, 'w') as f:
        f.write('')
    


# In[ ]:


while True:
    wait_to_read(VECTORS_COMPLETE)
    print("Simulations Complete")
    npz = np.load(VECTORS_NPZ)
    train_model(npz)
    os.remove(VECTORS_COMPLETE)
    os.remove(VECTORS_NPZ)

