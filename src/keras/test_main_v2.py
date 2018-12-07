import os
import datetime
import pickle
from utils.data_loading import make_folder_structure, load_data_to_folder # my python files

from keras.applications import VGG16
from keras import models
from keras import layers
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator

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

#  ###############################################
#  Initial Data handling #########################
#  ###############################################

base_dir = 'data'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')


main_classification_types = ['dyed-lifted-polyps',
                             'dyed-resection-margins',
                             'esophagitis',
                             'normal-cecum',
                             'normal-pylorus',
                             'normal-z-line',
                             'polyps',
                             'ulcerative-colitis']

# original data location path
orginal_data_folder = '/home/vajira/simula/Datasets/kvasir_v2_preprocessed_borders_navbox_removed'


###########################################################################################
no_of_classes = 8
train_size = 500
validation_size = 250
test_size = 250

img_height = 150
img_width = 150

batch_size = 25
no_of_epochs = 50

training_steps_per_epoch = (train_size * no_of_classes) / batch_size

validation_steps = (validation_size * no_of_classes) / batch_size

########################################################################################
print('trainig steps per epoch =', training_steps_per_epoch)
print('validation steps per epoch =', validation_steps)

########################################################################################


make_folder_structure(base_dir, main_classification_types)  # making the folder structure

# loading data to folders for trining, validation and testing
load_data_to_folder(orginal_data_folder, base_dir,
                    main_classification_types,
                    train_size, validation_size, test_size)

# the model directory
model_dir = 'my_models'
if not os.path.exists(model_dir):
    os.mkdir(model_dir, mode=0o777)

# history saving directory
history_dir = 'history_of_training'
if not os.path.exists(history_dir):
    os.mkdir(history_dir)

# plot directory
plot_dir = 'plots'
if not os.path.exists(plot_dir):
    os.mkdir(plot_dir)



#####################################################################################
#  # Load VGG16 base
conv_base = VGG16(weights='imagenet',
                  include_top=False,  # removing densely connected classifier
                  input_shape=(img_width, img_height, 3))  # this is optional

# ## Defining and training denesely connected classifier

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
#  model.add(layers.Dropout(0.5))
model.add(layers.Dense(8, activation='softmax')) ## change from sigmoid to softmax for 8 classes

model.summary()


# ##############################################################
#  Testing trainable weights

print('Trainable weights before freezing conv base=', len(model.trainable_weights))

conv_base.trainable = False  # Freezing the base model weights and bias

print('Trainable weights after freezing conv base:', len(model.trainable_weights))


#################################################################

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
    target_size=(img_width, img_height),  # resize all the images to 150X150
    batch_size=batch_size,
    class_mode='categorical')  # because loss in binary crossentropy

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

#  batch_size = 20 # in this case this should be a divisible of 1000 (because 1000 samples)

#################################################################

#################################################################
#  Model compilation


model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=2e-5),
              metrics=['acc'])

history = model.fit_generator(
    train_generator,
    steps_per_epoch=training_steps_per_epoch,  # 500*8/50 = 80
    epochs=no_of_epochs,
    validation_data=validation_generator,
    validation_steps=validation_steps)  # 250*8/50 = 40



model_name_string = 'medical_v1_' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M") + '.h5'
model_fname = os.path.join(model_dir, model_name_string)
model.save(model_fname)
print('Model saved')

#####################################################
### saving the training history for future graphing
#####################################################

history_string= 'history_of_medical_v1' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
history_fname = os.path.join(history_dir, history_string)
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

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-5),
              metrics=['acc'])

history = model.fit_generator(
    train_generator,
    steps_per_epoch=training_steps_per_epoch,
    epochs=no_of_epochs,
    validation_data=validation_generator,
    validation_steps=validation_steps)

#  Saving the model again (fined tunes)
model_name_string = 'medical_v1_fine_tunned' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M") + '.h5'
model_fname = os.path.join(model_dir, model_name_string)
model.save(model_fname)
print('Fined tunened Model saved')

#####################################################
### saving the training history for future graphing
#####################################################

history_sting= 'history_of_v6_medical_fine_tunned' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
history_fname = os.path.join(history_dir, history_sting)
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


print('Test OK')
