import numpy as np
import datetime
import os
import pickle
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.layers import Dropout, Flatten, Dense
from keras.applications import ResNet50
from keras.models import Model, Sequential
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.applications.resnet50 import preprocess_input
from keras import optimizers

import matplotlib as mpl

#  ######################################################
#  #### Matplotlib X display error - removing for server#
#  ######################################################
mpl.use('Agg')  # This has to run before pyplot import

import matplotlib.pyplot as plt

import sys
import pandas as pd



###################################################################
#  Getting main data directory
###############################################################

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

number_of_epochs = 10

batch_size = 32

trgt_sz = 224

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
    batch_size=batch_size, class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(validation_data_dir,
    shuffle=True,
    target_size=(trgt_sz, trgt_sz),
    batch_size=batch_size, class_mode='categorical')


#######################################################################
## Model ##############################################################
#######################################################################

base_model = ResNet50(weights='imagenet', include_top=True) # include top = True (previous case it was False)
x = Dense(16, activation=None, name='predictions')(base_model.layers[-2].output)


model = Model(inputs=base_model.input, output=x)

########################################################################

## Freez the base model #################

# for layer in base_model.layers: layer.trainable = False # all layers are trainable

#  Optimizer (SGD)
optimizer_sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

# Model compilation
model.compile(optimizer=optimizer_sgd , loss='categorical_crossentropy', metrics=['accuracy'])

print("Model compiled")

print("Start trainig...")
history = model.fit_generator(train_generator, train_generator.n // batch_size, epochs=number_of_epochs, workers=4,
        validation_data=validation_generator, validation_steps=validation_generator.n // batch_size)

print("Taining completed")

########################################################################

#  saving the model

model.save_weights(os.path.join(model_dir,model_name))

print("Model saved to", model_dir)

#####################################################################

# Plotting training history

def plot_training_history(history):
    train_acc = history.history['acc']
    val_acc = history.history['val_acc']
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    df = pd.DataFrame({'train_acc':train_acc, 'train_loss':train_loss, 'val_acc':val_acc, 'val_loss':val_loss})
    pie = df.plot()
    fig = pie.get_figure()
    fig.savefig(os.path.join(plot_dir, acc_loss_plot_name))

plot_training_history(history)

print("Plots saved to", plot_dir)

