#import numpy as np
#import datetime
import os
#import pickle
from keras.preprocessing.image import ImageDataGenerator
#from keras.preprocessing import image
#from keras.layers import Dropout, Flatten, Dense
from keras.applications import DenseNet201
from keras.models import Model, Sequential
from keras.layers import Dense, GlobalAveragePooling2D
#from keras import backend as K
from keras.applications.resnet50 import preprocess_input
from keras import optimizers
from keras.callbacks import ModelCheckpoint # to save and get only best model weights

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

number_of_epochs_pretraining = 5
number_of_epochs_posttraining = 5

batch_size = 16

trgt_sz = 224

checkpoint_name = "17_1_weights-improvement-{epoch:02d}-{val_acc:.4f}.hdf5"
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
    batch_size=batch_size, class_mode='categorical')  # changed from "categorical" to "binary"

validation_generator = test_datagen.flow_from_directory(validation_data_dir,
    shuffle=False,
    target_size=(trgt_sz, trgt_sz),
    batch_size=batch_size, class_mode='categorical')  # changed from "categorical" to "binary"


#######################################################################
## Model ##############################################################
#######################################################################

base_model = DenseNet201(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(16, activation='softmax')(x) # changing to detect only two types


model = Model(inputs=base_model.input, output=predictions)

########################################################################

## Freez the base model #################

for layer in base_model.layers: layer.trainable = False # all layers are  not trainable

#  Optimizer (SGD)
optimizer = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True) #optimizers.rmsprop()

# Model compilation
# Chaning loss function: categorical_crossentropy to binary_crossentropy
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

print("Model compiled")



print("Start the last layers trainig...")
history_1 = model.fit_generator(train_generator,
                                train_generator.n // batch_size,
                                epochs=number_of_epochs_pretraining,
                                workers=4,
                                validation_data=validation_generator,
                                validation_steps=validation_generator.n // batch_size)

print("Taining completed - last layers")

########################################################################
#  Starting training layres above 140 (last layers)
print("starting whole layres training of the model...")

#split_at = 140
#for layer in model.layers[:split_at]: layer.trainable = False
#for layer in model.layers[split_at:]: layer.trainable = True
for layer in base_model.layers: layer.trainable = True

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])


#######################################################################
# Looking for only the best model using validation accuracy
# checkpoint
filepath = os.path.join(model_dir, checkpoint_name) # name for the best model weights
# verbose = inoformation showing in training stage
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

#############################################################################

history_2 = model.fit_generator(train_generator,
                                train_generator.n // batch_size,
                                epochs=number_of_epochs_posttraining,
                                workers=4,
                                validation_data=validation_generator,
                                validation_steps=validation_generator.n // batch_size,
                                callbacks=callbacks_list)

########################################################################

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

    # writing plotting data to csv file
    df.to_csv(os.path.join(history_dir,plt_name), sep='\t', encoding='utf-8')

    pie = df.plot()
    fig = pie.get_figure()
    #fig.savefig(os.path.join(plot_dir, acc_loss_plot_name))
    fig.savefig(os.path.join(dir, plt_name))

plot_training_history(history_1, plot_dir, "history_1_pretraining" + accuracy_plot_name)
plot_training_history(history_2, plot_dir, "history_2_posttraining" + accuracy_plot_name)

print("Plots saved to", plot_dir)

