#  This is third test
#  Here, using the pre-trained network to classify medical Images
#  Modified for medical image classification

from keras.applications import VGG16
import os
import shutil
import datetime
# import numpy as np
from keras.preprocessing.image import ImageDataGenerator

import pickle # to save history file of training
from keras import models
from keras import layers
from keras import optimizers

import matplotlib as mpl
#  ######################################################
#  #### Matplotlib X display error - removing for server#
#  ######################################################
mpl.use('Agg')  # This has to run before pyplot import

import matplotlib.pyplot as plt

#  ##############################################
#  ### GPU memory cleaning - taken from a forum##
#  ##############################################
from keras import backend as k
cfg = k.tf.ConfigProto()
cfg.gpu_options.allow_growth = True
k.set_session(k.tf.Session(config=cfg))

#  ######################################################
#  ## Generating small data set from the original data set
#  ######################################################

original_dataset_dir = '/home/vajira/simula/Datasets/kvasir_v2_preprocessed_borders_navbox_removed'  # original data location path
data_dir = 'data'  # location to put sub samples of data (making small data set from large one)
base_dir = 'data/medical_image_v1'  # change this one to create new small datasets
model_dir = 'my_models'
plot_dir = 'plots'
samples_dir = 'samples'
extracted_features_dir = 'extracted_features'
training_history_dir = 'training_history'

range_start_point = 1000  # start point of the range of sample images to take sub samples from original data set

if not os.path.exists(training_history_dir):
    os.mkdir(training_history_dir)

if not os.path.exists(extracted_features_dir):
    os.mkdir(extracted_features_dir)

if not os.path.exists(data_dir):
    os.mkdir(data_dir)

if not os.path.exists(samples_dir):
    os.mkdir(samples_dir)

if not os.path.exists(plot_dir):
    os.mkdir(plot_dir)  # to save plots

if not os.path.exists(model_dir):
    os.mkdir(model_dir)  # to save models

if not os.path.exists(base_dir):
    os.mkdir(base_dir)  # new directory to add small data set

#  Main directories

# Train Directory
train_dir = os.path.join(base_dir, 'train')
if not os.path.exists(train_dir):
    os.mkdir(train_dir)

# Validation Directory
validation_dir = os.path.join(base_dir, 'validation')
if not os.path.exists(validation_dir):
    os.mkdir(validation_dir)

# Test Directory
test_dir = os.path.join(base_dir, 'test')
if not os.path.exists(test_dir):
    os.mkdir(test_dir)

# sub directories   ##################

#######################  Teaining directories ######################
'''
train__dir = os.path.join(train_dir, 'train_ _dir')
if not os.path.exists(train__dir):
    os.mkdir(train__dir)
'''

# Train dyed-lifted-polyps
train_dyed_lifted_polyps_dir = os.path.join(train_dir, 'dyed-lifted-polyps')
if not os.path.exists(train_dyed_lifted_polyps_dir):
    os.mkdir(train_dyed_lifted_polyps_dir)

# Train dyed-resection-margins Directory
train_dyed_resection_margins_dir = os.path.join(train_dir, 'dyed-resection-margins')
if not os.path.exists(train_dyed_resection_margins_dir):
    os.mkdir(train_dyed_resection_margins_dir)

# Train esophagitis Directory
train_esophagitis_dir = os.path.join(train_dir, 'esophagitis')
if not os.path.exists(train_esophagitis_dir):
    os.mkdir(train_esophagitis_dir)

# Train normal-cecum Directory
train_normal_cecum_dir = os.path.join(train_dir, 'normal-cecum')
if not os.path.exists(train_normal_cecum_dir):
    os.mkdir(train_normal_cecum_dir)

# Train normal-pylorus Directory
train_normal_pylorus_dir = os.path.join(train_dir, 'normal-pylorus')
if not os.path.exists(train_normal_pylorus_dir):
    os.mkdir(train_normal_pylorus_dir)

# Train normal-z-line Directory
train__dir = os.path.join(train_dir, 'train_ _dir')
if not os.path.exists(train__dir):
    os.mkdir(train__dir)

# Train polyps Directory

# Train ulcerative-colitis Directory


#######################################################

# Validation Cat Directory
validaton_cat_dir = os.path.join(validation_dir, 'cats')
if not os.path.exists(validaton_cat_dir):
    os.mkdir(validaton_cat_dir)

# Validation Dog Directory
validation_dog_dir = os.path.join(validation_dir, 'dogs')
if not os.path.exists(validation_dog_dir):
    os.mkdir(validation_dog_dir)

# Test Cat Directory
test_cat_dir = os.path.join(test_dir, 'cats')
if not os.path.exists(test_cat_dir):
    os.mkdir(test_cat_dir)

# Test Dog Directory
test_dogs_dir = os.path.join(test_dir, 'dogs')
if not os.path.exists(test_dogs_dir):
    os.mkdir(test_dogs_dir)
###########################################################
######
# copying first 1000 cat images to train_cats_dir (if the folder is empty)
fnames = ['cat.{}.jpg' .format(i) for i in range(range_start_point, range_start_point + 1000)]
for fname in fnames:
    srd = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_cats_dir, fname)
    if not os.path.exists(dst):
        shutil.copyfile(srd, dst)

# copying next 500 cat images to validation_cats_dir (if the folder is empty)
fnames = ['cat.{}.jpg' .format(i) for i in range(range_start_point + 1000, range_start_point + 1500)]
for fname in fnames:
    srd = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validaton_cat_dir, fname)
    if not os.path.exists(dst):
        shutil.copyfile(srd, dst)

# copying next 5oo cat images to test_cats_dir (if the folder is empty)
fnames = ['cat.{}.jpg' .format(i) for i in range(range_start_point + 1500, range_start_point + 2000)]
for fname in fnames:
    srd = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_cat_dir, fname)
    if not os.path.exists(dst):
        shutil.copyfile(srd, dst)

# copying first 1000 dog images to train_dogs_dir (if the folder is empty)
fnames = ['dog.{}.jpg' .format(i) for i in range(range_start_point, range_start_point + 1000)]
for fname in fnames:
    srd = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_dyed_resection_margins_dir, fname)
    if not os.path.exists(dst):
        shutil.copyfile(srd, dst)

# copying next 500 dog images to validation_dogs_dir (if the folder is empty)
fnames = ['dog.{}.jpg' .format(i) for i in range(range_start_point + 1000, range_start_point + 1500)]
for fname in fnames:
    srd = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_dog_dir, fname)
    if not os.path.exists(dst):
        shutil.copyfile(srd, dst)

# copying next 5oo dog images to test_dogs_dir (if the folder is empty)
fnames = ['dog.{}.jpg' .format(i) for i in range(range_start_point + 1500, range_start_point + 2000)]
for fname in fnames:
    srd = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_dogs_dir, fname)
    if not os.path.exists(dst):
        shutil.copyfile(srd, dst)


# printing data information
print('Total training cats= ', len(os.listdir(train_cats_dir)))
print('Total training dogs= ', len(os.listdir(train_dyed_resection_margins_dir)))
print('Total validation cats= ', len(os.listdir(validaton_cat_dir)))
print('Total validation dogs= ', len(os.listdir(validation_dog_dir)))
print('Total testing cats= ', len(os.listdir(test_cat_dir)))
print('Total testing dogs= ', len(os.listdir(test_dogs_dir)))

#####################################################################################
#  # Load VGG16 base
conv_base = VGG16(weights='imagenet',
                  include_top=False,  # removing densely connected classifier
                  input_shape=(150, 150, 3))  # this is optional

# ## Defining and training denesely connected classifier

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
#  model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()


# ##############################################################
#  Testing trainable weights

print('Trainable weights before freezing conv base=', len(model.trainable_weights))

conv_base.trainable = False  # Freezing the base model weights and bias

print('Trainable weights after freezing conv base:', len(model.trainable_weights))


#################################################################

train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1. / 255)

##############################################


train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),  # resize all the images to 150X150
    batch_size=20,
    class_mode='binary')  # because loss in binary crossentropy

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)

#  batch_size = 20 # in this case this should be a divisible of 1000 (because 1000 samples)

#################################################################
#  Model compilation

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=2e-5),
              metrics=['acc'])

history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=50)

# saving the model

model_name_string = 'cats_and_dogs_small_v6_medical' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M") + '.h5'
model_fname = os.path.join(model_dir, model_name_string)
model.save(model_fname)
print('Model saved')

#####################################################
### saving the training history for future graphing
#####################################################

history_sting= 'history_of_v6_medical' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
history_fname = os.path.join(training_history_dir, history_sting)
f = open(history_fname, "wb+")
pickle.dump(history.history, f)
f.close()
print('History of the model saved')

############################################################################################
# ## Plotting data to understand and save plots to plot folder ###############
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc)+1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
title_string = 'Training and validation accuracy =' + model_name_string
plt.title(title_string)
plt.legend()

# To save plots - added by me
plot_name = title_string + '.tiff'
plot_fname = os.path.join(plot_dir, plot_name)
plt.savefig(plot_fname)
##############################

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
title_string = 'Training and validation loss' + model_name_string
plt.title(title_string)
plt.legend()

# To save plots - added by me
plot_name = title_string + '.tiff'
plot_fname = os.path.join(plot_dir, plot_name)
plt.savefig(plot_fname)



###########################################################################################
#  Starting Fine tuning
###########################################################################################
print('starting fine tuning')

conv_base.summary()

conv_base.trainable = True

set_trainable = False
for layer in conv_base.layers:
    print(layer.name)
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

############################################################################################
#  Recompiling with unfreez layers

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-5),
              metrics=['acc'])

history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=50)

#  Saving the model again (fined tunes)
model_name_string = 'cats_and_dogs_small_v6_medical_fine_tunned' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M") + '.h5'
model_fname = os.path.join(model_dir, model_name_string)
model.save(model_fname)
print('Fined tunened Model saved')

#####################################################
### saving the training history for future graphing
#####################################################

history_sting= 'history_of_v6_medical_fine_tunned' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
history_fname = os.path.join(training_history_dir, history_sting)
f = open(history_fname, "wb+")
pickle.dump(history.history, f)
f.close()
print('History of the model saved')


############################################################################################
# ## Plotting data to understand and save plots to plot folder ###############

plt.figure()

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc)+1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
title_string = 'Training and validation accuracy =' + model_name_string
plt.title(title_string)
plt.legend()

# To save plots - added by me
plot_name = title_string + '.tiff'
plot_fname = os.path.join(plot_dir, plot_name)
plt.savefig(plot_fname)
##############################

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
title_string = 'Training and validation loss' + model_name_string
plt.title(title_string)
plt.legend()

# To save plots - added by me
plot_name = title_string + '.tiff'
plot_fname = os.path.join(plot_dir, plot_name)
plt.savefig(plot_fname)


############# Smooth plotting ##############################################################
# to implement from book or use good new tools for plotting
#############################################################################################
