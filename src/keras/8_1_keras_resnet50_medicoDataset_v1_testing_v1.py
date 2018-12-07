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
import numpy as np
import itertools


from sklearn.metrics import confusion_matrix



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



model_name = arg_model_name # "8_1_keras_resnet50_v2" # take my file name as the model name

cm_plot_name = 'cm_'+model_name


batch_size = 64

trgt_sz = 224


#########################################
#  Managing Directory
#########################################
if not os.path.exists(plot_dir):
    os.mkdir(plot_dir)

#  Here - Test directory is same as validation directory

test_data_dir = f'{data_dir}/validation'

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

test_generator = test_datagen.flow_from_directory(test_data_dir,
    shuffle=False,
    target_size=(trgt_sz, trgt_sz),
    batch_size=batch_size, class_mode='categorical')

####################################################################
#  Making the model structure to load weights
##################################################################


base_model = ResNet50(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(16, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

##############################################################

model.load_weights(os.path.join(model_dir, model_name))

print("Modle loaded succesfully")

############################################################

# start perdictions

probabilities = model.predict_generator(test_generator)

predicted = np.argmax(probabilities.data,1) # take predicted indices

#  Generate confusion matrix
cm = confusion_matrix(test_generator.classes, predicted)

# take class names to plot confusion matrix
class_names = test_generator.class_indices
class_names = list(class_names.keys())
class_names = np.asarray(class_names)

#############################################################

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          plt_size=[10,10]):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.rcParams['figure.figsize'] = plt_size
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(plot_dir, cm_plot_name))


####################################################################

# plot and save confusion matrix
plot_confusion_matrix(cm, classes=class_names, title='my confusion matrix')

print("confusion matrix plot has been saved to directory =", plot_dir)
