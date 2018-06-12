
# coding: utf-8

# # Imports

# In[ ]:


import keras
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.io import wavfile

from sklearn.metrics import accuracy_score, hamming_loss, zero_one_loss, auc

from bokeh.plotting import figure, show
from bokeh.io import output_notebook

from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
import random

from keras.layers import Conv1D, MaxPool1D, Activation, Dense, Input, Flatten, BatchNormalization, Dropout
from keras.losses import binary_crossentropy
from keras.optimizers import SGD
from keras.utils import Sequence
import keras.backend as K
import tensorflow as tf


# ###### Parameters

# In[ ]:


#Hardware Parameters
n_gpus = 4

#Training Parameters
batch_size = 32
max_epochs = 200
max_trainings = 10
kernel_initializer = 'he_uniform'

# SGD parameters
starting_learning_rate = 0.01
momentum = 0.9
global_decay = 0.2
local_decay = 1e-6

# EarlyStopping Parameters
min_improvement = 0
patience = 5

# Paths
dataset_dir = '../data/MagnaTagATune/rawwav_2/'
annotations_path = '../data/MagnaTagATune/annotation_reduced.csv'

checkpoint_dir = './checkpoints_3/'
checkpoint_file_name = 'weights-{epoch:03d}-{val_loss:.5f}.hdf5'
log_dir ='./logs'


# # Functions

# ###### Data reading during training

# In[ ]:


class MagnaTagATuneSequence(Sequence):

    def __init__(self, train_set_paths, train_set_labels, batch_size):
        self.paths, self.y = train_set_paths, train_set_labels
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.paths) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x_paths = self.paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = []
        for value in batch_x_paths:
            path = dataset_dir + value[:-3]+'wav'
            _, data = wavfile.read(path)
            batch_x.append(data)
        batch_x = np.array(batch_x)[:,:,np.newaxis]        
        return (batch_x,batch_y)    


# ###### Performance Metrics (not used anymore)

# In[ ]:


def ratio_wrong_over_correct_ones(y_true, y_pred):
    op1 = K.sum(K.abs(K.cast(y_true - K.round(y_pred), dtype='float32')))
    op2 = K.sum(K.cast(K.equal(y_true,1.0),dtype='float32'))
    return op1/op2

def ratio_correct_ones(y_true, y_pred):
    op1 = K.sum(K.cast(K.equal(y_true + K.round(y_pred),2.0),dtype='float32'))
    op2 = K.sum(K.cast(K.equal(y_true,1.0),dtype='float32'))
    return op1/op2

def auc_roc(y_true, y_pred):
    # any tensorflow metric
    value, update_op = tf.metrics.auc(y_true, y_pred, summation_method='careful_interpolation')

    # find all variables created for this metric
    metric_vars = [i for i in tf.local_variables() if 'auc_roc' in i.name.split('/')[1]]

    # Add metric variables to GLOBAL_VARIABLES collection.
    # They will be initialized for new session.
    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

    # force to update metric values
    with tf.control_dependencies([update_op]):
        value = tf.identity(value)
        return value


# ###### Best checkpoint selection

# In[ ]:


def find_best_checkpoint(prev_chkpts):
    best_ratio = np.inf
    best_chkpt = ''
    best_epoch = 0
    for chkpt in prev_chkpts:
        epoch = int(chkpt[8:11])
        ratio = float(chkpt[12:19])
        
        if ratio < best_ratio:
            best_ratio = ratio
            best_chkpt = chkpt
            best_epoch = epoch
    print('\n starting from model {} \n'.format(best_chkpt))
    return best_chkpt, best_epoch


# # Preparation

# ###### Prepare Training and Validation  Sets

# In[ ]:


annotations = pd.read_csv(annotations_path, sep='\t')
t_size = 0.71774

train_set, test_set = train_test_split(annotations['mp3_path'], train_size=t_size, test_size=(1-t_size)) 
test_set, val_set = train_test_split(test_set, train_size=0.75, test_size=0.25) 
#train_set= train_set.loc[train_set.str.len()<70]
#test_set= test_set.loc[test_set.str.len()<70]

train_set_paths = train_set.values
train_set_labels = annotations.loc[annotations['mp3_path'].isin(train_set)].drop(columns=['mp3_path','Unnamed: 0']).values
train_set_size = len(train_set_paths)
print("Train set size: {} \n".format(train_set_size))


y_dimension = train_set_labels.shape[1]

_, data = wavfile.read( dataset_dir + annotations['mp3_path'][0][:-3]+ 'wav')
x_dimension = len(data)

print("X dimension: {}\nY dimension: {} \n".format(x_dimension, y_dimension))

val_set_paths = val_set.values
val_set_labels = annotations.loc[annotations['mp3_path'].isin(val_set)].drop(columns=['mp3_path','Unnamed: 0']).values
val_set_size = len(val_set_paths)
print("Validation set size: {} \n".format(val_set_size))


# In[ ]:


print('\n * * * Loading Validation Set * * * \n')

val_set_data = []
for value in tqdm(val_set_paths):
    path = dataset_dir+value[:-3]+'wav'
    _, data = wavfile.read(path)
    val_set_data.append(data)  
val_set_data = np.array(val_set_data)[:,:,np.newaxis] 

print('\n * * * Done * * * \n')


# ###### Modify session

# In[ ]:


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
keras.backend.set_session(session)


# ######  Building Model

# In[ ]:


model = keras.Sequential()

model.add(Conv1D(filters=128, kernel_size=3, strides=3, padding='valid', input_shape=(x_dimension,1), kernel_initializer=kernel_initializer))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv1D(filters=128, kernel_size=3, strides=1, padding='same', kernel_initializer=kernel_initializer))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPool1D(pool_size=3))

model.add(Conv1D(filters=256, kernel_size=3, strides=1, padding='same', kernel_initializer=kernel_initializer))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPool1D(pool_size=3))

model.add(Conv1D(filters=256, kernel_size=3, strides=1, padding='same', kernel_initializer=kernel_initializer))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPool1D(pool_size=3))

model.add(Conv1D(filters=256, kernel_size=3, strides=1, padding='same', kernel_initializer=kernel_initializer))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPool1D(pool_size=3))

model.add(Conv1D(filters=256, kernel_size=3, strides=1, padding='same', kernel_initializer=kernel_initializer))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPool1D(pool_size=3))

model.add(Conv1D(filters=256, kernel_size=3, strides=1, padding='same', kernel_initializer=kernel_initializer))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPool1D(pool_size=3))

model.add(Conv1D(filters=256, kernel_size=3, strides=1, padding='same', kernel_initializer=kernel_initializer))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPool1D(pool_size=3))

model.add(Conv1D(filters=512, kernel_size=3, strides=1, padding='same', kernel_initializer=kernel_initializer))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPool1D(pool_size=3))

model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(units=y_dimension, activation='sigmoid'))


# In[ ]:


model.summary()


# ###### Callbacks definition

# In[ ]:


class MyCallBack(keras.callbacks.Callback):
    def __init__(self, callbacks, model, is_tb=False):
            super().__init__()
            self.callback = callbacks
            self.is_tb = is_tb
            if not self.is_tb:
                self.model = model
                self.model_original = model

    def on_epoch_begin(self,epoch,logs=None):
            if not self.is_tb:
                self.model = self.model_original
            self.callback.on_epoch_begin(epoch, logs=logs)

    def on_epoch_end(self,epoch,logs=None):
            if not self.is_tb:
                self.model = self.model_original
            else:
                y_pred = self.model.predict(self.validation_data[0])
                auc_skl = roc_auc_score(self.validation_data[1], y_pred)
                print('\nSKLearn validation auc: {}'.format(auc_skl))
            self.callback.on_epoch_end(epoch, logs=logs)

    def on_batch_end(self, batch, logs=None):
            if not self.is_tb:
                self.model = self.model_original
            self.callback.on_batch_end(batch, logs=logs)

    def on_batch_begin(self, batch, logs=None):
            if not self.is_tb:
                self.model = self.model_original
            self.callback.on_batch_begin(batch, logs=logs)

    def on_train_begin(self, logs=None):
            if not self.is_tb:
                self.model = self.model_original
            self.callback.set_model(self.model)
            self.callback.on_train_begin(logs=logs)

    def on_train_end(self, logs=None):
            if not self.is_tb:
                self.model = self.model_original
            self.callback.on_train_end(logs=logs)

cbk_tb = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, batch_size=batch_size, write_graph=True,
                                         write_grads=False, write_images=False, embeddings_freq=0,
                                         embeddings_layer_names=None, embeddings_metadata=None)

cbk_es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min',
                                          min_delta=min_improvement, patience=patience, verbose=1)

cbk_mc = keras.callbacks.ModelCheckpoint(monitor='val_loss', mode='min', save_best_only=True, 
                                            filepath=checkpoint_dir+checkpoint_file_name, 
                                            verbose=1)

cbk = MyCallBack(cbk_tb, model, is_tb=True)
cbk1 = MyCallBack(cbk_es, model)
cbk2 = MyCallBack(cbk_mc, model)

callbacks = [cbk,cbk1,cbk2]


# ### Training

# In[ ]:


initial_epoch = 0
training_nr = 0

parallel_model = keras.utils.multi_gpu_model(model, gpus=n_gpus)

while (initial_epoch <= max_epochs) and (training_nr <= max_trainings):
    print('\n\n* * * * Starting training {0} from epoch {1} * * * * \n\n'.format(training_nr,  initial_epoch+1))
    best_checkpoint = ''
    best_epoch = 0
    
    previous_checkpoints = os.listdir(checkpoint_dir)
    
    if previous_checkpoints != []:
        best_checkpoint, best_epoch = find_best_checkpoint(previous_checkpoints)
        initial_epoch = best_epoch      
       
           
    decay = global_decay ** training_nr
    learning_rate = starting_learning_rate * decay        
                
        
    training_nr = training_nr + 1
    
    optimizer = SGD(lr = learning_rate, momentum=momentum, decay=local_decay , nesterov=True)
    
    if len(previous_checkpoints)!=0:
        model.load_weights(checkpoint_dir + best_checkpoint)
        parallel_model = keras.utils.multi_gpu_model(model, gpus=n_gpus)
    
    
    parallel_model.compile(optimizer=optimizer, loss='binary_crossentropy',)
    
    parallel_model.fit_generator(MagnaTagATuneSequence(train_set_paths, train_set_labels, batch_size),
                                #validation_data = MagnaTagATuneSequence(val_set_paths, val_set_labels, batch_size),
                                 validation_data = (val_set_data, val_set_labels),
                                 epochs=max_epochs, callbacks = callbacks, initial_epoch = initial_epoch)


# ###### Prepare Test Set

# In[ ]:


test_set_paths = test_set.values
test_set_labels = annotations.loc[annotations['mp3_path'].isin(test_set)].drop(columns=['mp3_path','Unnamed: 0']).values
test_set_size = len(test_set_paths)
print("Test set size: {} ".format(test_set_size))


# test_set_data = []
# for value in tqdm(test_set_paths):
#     path = '../data/MagnaTagATune/rawwav_2/'+value[:-3]+'wav'
#     _, data = wavfile.read(path)
#     test_set_data.append(data)  
# test_set_data = np.array(test_set_data)[:,:,np.newaxis]  

# ###### Load best Model

# In[ ]:


previous_checkpoints = os.listdir(checkpoint_dir)
best_checkpoint, best_epoch = find_best_checkpoint(previous_checkpoints)
model.load_weights(checkpoint_dir + best_checkpoint)
#model = keras.models.load_model(checkpoint_dir + best_checkpoint)
parallel_model = keras.utils.multi_gpu_model(model, gpus=n_gpus)


# ###### Prediction and evaluation

# In[ ]:


predictions = parallel_model.predict_generator(MagnaTagATuneSequence(test_set_paths, test_set_labels, batch_size), verbose=1)
#predictions = parallel_model.predict(test_set_data,batch_size=batch_size,verbose=1)


# In[ ]:


try:
    roc_auc = roc_auc_score(test_set_labels, predictions)
    print("Test roc auc result: {} ".format(roc_auc))
except ValueError:
    print('ERROR ON TEST ROC')


# In[ ]:

