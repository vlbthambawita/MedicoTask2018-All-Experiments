import os
import mxnet as mx
import numpy as np
import sys

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

data_dir = arg_main_data_dir #"../../data/data_v2" #main_data_dir
model_name = arg_model_name

model_dir = data_dir + '/mxnet_models'
plot_dir  = data_dir + '/mxnet_plots'
history_dir = data_dir + '/mxnet_history'

train_data_dir = f'{data_dir}/train'
validation_data_dir = f'{data_dir}/validation'


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


dataset_train = ImageFolderDataset(root=train_data_dir,transform=transform)
dataset_test = ImageFolderDataset(root=validation_data_dir,transform=transform)

dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=NUM_WORKERS) # last_batch='discard' (removed for testing)
dataloader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE, # last_batch='discard',
                             shuffle=True, num_workers=NUM_WORKERS)
print("Train dataset: {} images, Test dataset: {} images".format(len(dataset_train), len(dataset_test)))


################################################
#  Check categories - for debuging only
################################################

categories = dataset_train.synsets
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
#  Evaluation accuracy
#################################################

def evaluate_accuracy_gluon(data_iterator, net):
    num_instance = nd.zeros(1, ctx=ctx)
    sum_metric = nd.zeros(1,ctx=ctx, dtype=np.int32)
    for i, (data, label) in enumerate(data_iterator):
        data = data.astype(np.float32).as_in_context(ctx)
        label = label.astype(np.int32).as_in_context(ctx)
        output = net(data)
        prediction = nd.argmax(output, axis=1).astype(np.int32)
        num_instance += len(prediction)
        sum_metric += (prediction==label).sum()
    accuracy = (sum_metric.astype(np.float32)/num_instance.astype(np.float32))
    return accuracy.asscalar()


##################################################
#  Check validation accuracy before training
##################################################

print("Untrained network Test Accuracy: {0:.4f}".format(evaluate_accuracy_gluon(dataloader_test, my_net)))



##################################################
##################################################
#  Training
##################################################
##################################################



##################################################
# Set trainining parameters
##################################################

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

LEARNING_RATE = 0.0005
WDECAY = 0.00001
MOMENTUM = 0.9

trainer = gluon.Trainer(my_net.collect_params(), 'sgd',
                        {'learning_rate': LEARNING_RATE,
                         'wd':WDECAY,
                         'momentum':MOMENTUM})

##################################################
#  Start Training
##################################################
print('Start Training....')

val_accuracy = 0
for epoch in range(NO_OF_EPOCHS):
    for i, (data, label) in enumerate(dataloader_train):
        data = data.astype(np.float32).as_in_context(ctx)
        label = label.as_in_context(ctx)

        if i%20==0 and i >0:
            print('Batch [{0}] loss: {1:.4f}'.format(i, loss.mean().asscalar()))

        with autograd.record():
            output = my_net(data)
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        trainer.step(data.shape[0])

    nd.waitall() # wait at the end of the epoch
    new_val_accuracy = evaluate_accuracy_gluon(dataloader_test, my_net)
    print("Epoch [{0}] Test Accuracy {1:.4f} ".format(epoch, new_val_accuracy))

    # We perform early-stopping regularization, to prevent the model from overfitting
    if val_accuracy > new_val_accuracy:
        print('Validation accuracy is decreasing, stopping training')
        break
    val_accuracy = new_val_accuracy


print('Training finished..')


#######################################################
#  Saving the model
#######################################################

print('Saving model..')
my_net.save_parameters(os.path.join(model_dir,model_name))
print('Model saved to=', model_dir)


#######################################################



