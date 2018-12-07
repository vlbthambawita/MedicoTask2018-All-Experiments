#import numpy as np
#import datetime
import os
#import pickle
from keras.preprocessing.image import ImageDataGenerator
#from keras.preprocessing import image
#from keras.layers import Dropout, Flatten, Dense
from keras.applications import ResNet50
from keras.models import Model, Sequential
from keras.layers import Dense, GlobalAveragePooling2D
#from keras import backend as K
from keras.applications.resnet50 import preprocess_input
from keras import optimizers
from keras.callbacks import ModelCheckpoint # to save and get only best model weights
from keras import models
from keras import layers

import matplotlib as mpl

#  ######################################################
#  #### Matplotlib X display error - removing for server#
#  ######################################################
mpl.use('Agg')  # This has to run before pyplot import

# import matplotlib.pyplot as plt

import sys
import pandas as pd


# #######################################################
#  Getting main data directory
# #########################################################

arg_main_data_dir = sys.argv[1]  # Main data directory to be handled
arg_model_name = sys.argv[2] # model name to be saves

#my_file_name = model_name #"8_1_pytorch_resnet18_v1"  # model name to be saved

###########################################

#  Set parameters here
data_dir = arg_main_data_dir #"../../../data/data_generated_medicotask_v1" #main_data_dir
model_dir = data_dir + '/keras_models'
plot_dir  = data_dir + '/keras_plots'
history_dir = data_dir + '/keras_history'


model_name = arg_model_name #"8_1_keras_resnet50_v2" # take my file name as the model name

acc_loss_plot_name = 'acc_loss_plot_' + model_name
accuracy_plot_name = 'accuracy_plot_' + model_name
loss_plot_name = 'loss_plot_' + model_name

number_of_epochs= 5
#number_of_epochs_posttraining = 5

batch_size = 32

trgt_sz = 224

checkpoint_name = "15_1_weights-improvement-{epoch:02d}-{val_acc:.4f}.hdf5"
#'model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'

########################################################################
#  Managin Directory structure
########################################################################
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

if not os.path.exists(plot_dir):
    os.mkdir(plot_dir)

if not os.path.exists(history_dir):
    os.mkdir(history_dir)


#####################################################################

train_data_dir = f'{data_dir}/train'
validation_data_dir = f'{data_dir}/validation'

#  Preparing Data - Training and Validation

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
    shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(train_data_dir,
    shuffle=True,
    target_size=(trgt_sz, trgt_sz),
    batch_size=batch_size, class_mode='binary')  # changed from "categorical" to "binary"

validation_generator = test_datagen.flow_from_directory(validation_data_dir,
    shuffle=False,
    target_size=(trgt_sz, trgt_sz),
    batch_size=batch_size, class_mode='binary')  # changed from "categorical" to "binary"


#######################################################################
## Model ##############################################################
#######################################################################
'''
base_model = ResNet50(weights=None, include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x) # changing to detect only two types

'''

simple_model = models.Sequential()
simple_model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(trgt_sz, trgt_sz, 3)))
simple_model.add(layers.MaxPool2D((2,2)))
simple_model.add(layers.Conv2D(64, (3,3), activation='relu'))
simple_model.add(layers.MaxPool2D((2,2)))
simple_model.add(layers.Conv2D(64, (3,3), activation='relu'))
simple_model.add(layers.MaxPool2D((2,2)))
simple_model.add(layers.Conv2D(128, (3,3), activation='relu'))
simple_model.add(layers.MaxPool2D((2,2)))
simple_model.add(layers.Conv2D(128, (3,3), activation='relu'))
simple_model.add(layers.MaxPool2D((2,2)))
simple_model.add(layers.Conv2D(128, (3,3), activation='relu'))
simple_model.add(layers.Flatten())
simple_model.add(layers.Dense(64, activation='relu'))
simple_model.add(layers.Dense(1, activation='sigmoid'))


# model = Model(inputs=base_model.input, output=predictions)

########################################################################
print("Model summary = ", simple_model.summary())
####################################################

# exit()
## Freez the base model #################

# for layer in base_model.layers: layer.trainable = False # all layers are  not trainable

#  Optimizer (SGD)
optimizer = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True) #optimizers.rmsprop()
# optimizer = optimizers.rmsprop()

# Model compilation
# Chaning loss function: categorical_crossentropy to binary_crossentropy
#model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
simple_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

print("Model compiled")







########################################################################

# Looking for only the best model using validation accuracy
# checkpoint
filepath = os.path.join(model_dir, checkpoint_name) # name for the best model weights
# verbose = inoformation showing in training stage
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

#############################################################################
print("Start trainig...")

# exit()
history = simple_model.fit_generator(train_generator,
                                train_generator.n // batch_size,
                                epochs=number_of_epochs,
                                workers=4,
                                validation_data=validation_generator,
                                validation_steps=validation_generator.n // batch_size,
                                callbacks=callbacks_list)

########################################################################

print("Training completed")
#  saving the model

# model.save_weights(os.path.join(model_dir,model_name))

print("Best Model saved to", model_dir)

#####################################################################

# Plotting training history

# history = history_1.append(history_2)

def plot_training_history(history, dir, plt_name):
    train_acc = history.history['acc']
    val_acc = history.history['val_acc']
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    df = pd.DataFrame({'train_acc':train_acc, 'train_loss':train_loss, 'val_acc':val_acc, 'val_loss':val_loss})
    pie = df.plot()
    fig = pie.get_figure()
    #fig.savefig(os.path.join(plot_dir, acc_loss_plot_name))
    fig.savefig(os.path.join(dir, plt_name))

plot_training_history(history, plot_dir, "history" + accuracy_plot_name)


print("Plots saved to", plot_dir)

