import os
import mxnet as mx
import numpy as np
import sys
from sklearn.metrics import confusion_matrix
import pandas as pd
import itertools
import matplotlib.pyplot as plt

from mxnet.gluon.data.vision.datasets import ImageFolderDataset
from mxnet.gluon.data import DataLoader
from mxnet.gluon import nn
from mxnet.gluon.model_zoo import vision as models
from mxnet import gluon, nd, autograd

###################################################################
#  Getting input arguments
###############################################################

arg_main_data_dir = sys.argv[1]  # Main data directory to be handled
arg_model_name = sys.argv[2] # model name to be saves




##############################################
#  Data folders
##############################################

best_model_weights = input('Please, enter best weights value file name:')#"mxnet-resnet50-weights-improvement-02-0.8931.hdf5"

data_dir = arg_main_data_dir #"../../data/data_v2" #main_data_dir
model_name = arg_model_name

model_dir = data_dir + '/mxnet_models'
plot_dir  = data_dir + '/mxnet_plots'
history_dir = data_dir + '/mxnet_history'

cm_plot_name = 'cm_'+model_name

test_data_dir = f'{data_dir}/test'


#############################################
#  Managing Directory structure
############################################


if not os.path.exists(model_dir):
    os.mkdir(model_dir)

if not os.path.exists(plot_dir):
    os.mkdir(plot_dir)

if not os.path.exists(history_dir):
    os.mkdir(history_dir)



###############################################
#  Parameters
###############################################

NO_OF_EPOCHS = 2
TARGET_SIZE = 224
SIZE = (TARGET_SIZE, TARGET_SIZE)
BATCH_SIZE = 32
NUM_WORKERS = 4  # multiprocessing.cpu_count() - # to check

#  Check GPU availability and run on GPUs
ctx = mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()


###############################################
#  Transform function
###############################################


def transform(image, label):

    # resize the shorter edge to 224, the longer edge will be greater or equal to 224
    resized = mx.image.resize_short(image, TARGET_SIZE)

    # center and crop an area of size (224,224)
    cropped, crop_info = mx.image.center_crop(resized, SIZE)

    # transpose the channels to be (3,224,224)
    transposed = nd.transpose(cropped, (2, 0, 1))

    return transposed, label


################################################
#  Loading Images from folders
################################################

dataset_test = ImageFolderDataset(root=test_data_dir, transform=transform)
dataloader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE, # last_batch='discard',
                             shuffle=True, num_workers=NUM_WORKERS)
print("Test dataset: {} images".format(len(dataset_test)))


################################################
#  Check categories - for debuging only
################################################

categories = dataset_test.synsets
NUM_CLASSES = len(categories)

print(categories)
print(NUM_CLASSES)


################################################
################################################
#  Model creation
################################################

# get pretrained squeezenet
net = models.resnet50_v2(pretrained=True, prefix='kavisar',ctx=ctx)
my_net = models.resnet50_v2(prefix='kavisar', classes=8, ctx=ctx)
my_net.collect_params().initialize(ctx=ctx)
my_net.features = net.features


#################################################
#  Loading values to the models
#################################################

my_net.load_parameters(os.path.join(model_dir, best_model_weights))



##################################################
#  Tesing the model
##################################################

acc = mx.metric.Accuracy()

all_true_labels = np.array([])  # mx.nd.array([0])
# temp = mx.nd.array([1,2,3,4,5])
# all_predicted_labels = []
all_predicted_labels_device = mx.nd.array([-1], ctx=ctx)  # mx.nd

for i, (data, label) in enumerate(dataloader_test):
    data = data.astype(np.float32).as_in_context(ctx)  # loading data to GPU if available
    l = label.asnumpy()
    # label = label #.as_in_context(ctx) # loading data to GPU if available
    # all_true_labels = mx.ndarray.concat(all_true_labels,l, dim=0 )
    all_true_labels = np.concatenate((all_true_labels, l))
    print(l)
    print("====")
    print(label)
    print("====")
    print(len(all_true_labels))

    with autograd.predict_mode():
        probability = my_net(data)
        predictions = nd.argmax(probability, axis=1)
        all_predicted_labels_device = mx.ndarray.concat(all_predicted_labels_device, predictions, dim=0)
        print(predictions)
        acc.update(preds=predictions, labels=label)
        #  print(acc.get()[1])
        # all_true_labels.extend(label)

        # all_predicted_labels.extend(predictions)
        print("gpu array =", all_predicted_labels_device)
    #  print(label)


########################################################
#  Analyzing the results
########################################################

# take gpu array to the cpu
all_predicted_labels_cpu = all_predicted_labels_device.as_in_context(mx.cpu())

# remove the first element of array (it was added to resolve merging problem of NDarray)
all_predicted_labels_cpu = all_predicted_labels_cpu[1:].asnumpy()

#  Generate confusion matrix
cm = confusion_matrix(all_true_labels, all_predicted_labels_cpu)


######################################################
#  Confusion matrix plot fuction
######################################################
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


#######################################################
# Plotting and saving confusion matrix
######################################################
plot_confusion_matrix(cm, classes=categories, title='my confusion matrix')


print("confusion matrix plot has been saved to = ", plot_dir)

print("Testing Done... !!!")


#######################################################
#  Saving the model
#######################################################

#print('Saving model..')
#my_net.save_parameters(os.path.join(model_dir, model_name))
#print('Model saved to=', model_dir)


#######################################################



