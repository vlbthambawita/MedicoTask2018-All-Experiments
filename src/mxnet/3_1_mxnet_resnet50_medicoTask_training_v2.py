'''
This is trying to improve performance of training by using modifications to performance matric
gathering method - using NDARRAYS in GPUs
'''

import os
import mxnet as mx
import numpy as np
import sys
import pandas as pd
import copy

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

acc_loss_plot_name = 'acc_loss_plot_' + model_name

train_data_dir = f'{data_dir}/train'
validation_data_dir = f'{data_dir}/validation'

checkpoint_name = "deepcopy_mxnet-resnet50-weights-improvement-{epoch:02d}-{val_acc:.4f}.hdf5"
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

NO_OF_EPOCHS = 10
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
dataset_train = ImageFolderDataset(root=train_data_dir, transform=transform)
dataset_test = ImageFolderDataset(root=validation_data_dir, transform=transform)

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
my_net = models.resnet50_v2(prefix='kavisar', classes=NUM_CLASSES, ctx=ctx)
my_net.collect_params().initialize(ctx=ctx)
my_net.features = net.features



########################################################
#  New evaluate accuracy method
#  This should be optimized for GPU (making nd array in GPU)
########################################################


def evaluate_accuracy_and_loss(data_iterator, loss_fn, net):
    metric_acc = mx.metric.Accuracy()
    #  numerator = 0.
    #  denominator = 0.
    #  cumulative_loss = 0.
    #  no_of_samples = 0
    cumulative_loss = mx.nd.zeros(1, ctx=ctx)
    no_of_samples = mx.nd.zeros(1, ctx=ctx)

    for i, (data, label) in enumerate(data_iterator):

        #with autograd.predict_mode():

        data = data.astype(np.float32).as_in_context(ctx)
        label = label.astype(np.int32).as_in_context(ctx)
        output = net(data)
        loss = loss_fn(output, label)
        prediction = nd.argmax(output, axis=1).astype(np.int32)
        cumulative_loss += nd.sum(loss).asscalar()
        no_of_samples += data.shape[0]

        metric_acc.update([label], [prediction])
        # metric_loss.update([label], [prediction])

    print("cumulative loss = {0} no_of_samples = {1}".format(cumulative_loss.asscalar(), no_of_samples.asscalar()))
    loss = cumulative_loss.asscalar() / no_of_samples.asscalar()

    return metric_acc.get()[1], loss


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
#  Check validation accuracy and loss before training
##################################################


#acc_before, loss_before = evaluate_accuracy_and_loss(dataloader_test, softmax_cross_entropy, my_net)

#print("Untrained network Test Accuracy: {0:.4f}".format(evaluate_accuracy_gluon(dataloader_test, my_net)))

#print("Untrained network Test Accuracy: {0:.4f} Test Loss: {1:.4f} ".format(acc_before, loss_before))

##################################################
##################################################
#  Training
##################################################
##################################################


def train_model(net, trainer, loss_fn, num_epochs):

    best_val_accuracy = mx.nd.zeros(1, ctx=ctx)
    # df = pd.DataFrame(columns=['train_acc', 'train_loss', 'val_acc', 'val_loss'])
    history = mx.nd.empty((num_epochs, 4), ctx=ctx)  # 4 represents = train_acc, train_loss, val_acc, val_loss
    best_model = copy.deepcopy(net) # reference variable to best model
    best_model_name = "" # reference variable to best model name

    for epoch in range(num_epochs):

        print("Runing epoch = {0}".format(epoch))

        for i, (data, label) in enumerate(dataloader_train):
            data = data.astype(np.float32).as_in_context(ctx)
            label = label.as_in_context(ctx)

            #if i % 20 == 0 and i > 0:
             #   print('Batch [{0}] loss: {1:.4f}'
             #         .format(i, loss.mean().asscalar()))

            with autograd.record():
                output = net(data)
                loss = softmax_cross_entropy(output, label)
                loss.backward()
                print(i, sep='-', end='=', flush=True) # just for testing purpose
            trainer.step(data.shape[0])

        nd.waitall()  # wait at the end of the epoch

        #train_accuracy, train_loss = evaluate_accuracy_and_loss(dataloader_train, loss_fn, net)
       # new_val_accuracy, new_val_loss = evaluate_accuracy_and_loss(dataloader_test, loss_fn, net)
        history[epoch, 0], history[epoch, 1] = evaluate_accuracy_and_loss(dataloader_train, loss_fn, net)
        history[epoch, 2], history[epoch, 3] = evaluate_accuracy_and_loss(dataloader_test, loss_fn, net)
       # print("metrix loaded...")
       # new_val_accuracy = history[epoch, 2].asscalar()
       # print("new val accuracy found")
        #df2 = pd.DataFrame([[train_accuracy, train_loss, new_val_accuracy, new_val_loss]],
        #                   columns=['train_acc', 'train_loss', 'val_acc', 'val_loss'])
        # new_val_accuracy = evaluate_accuracy_gluon(dataloader_test, my_net)
       # print("all done")
        #print(type(train_accuracy))
      #  print("Epoch [{0}] Train accuracy {1:.4f} val Accuracy {2:.4f} "
            #  .format(epoch, history[epoch, 0].asscalar(), history[epoch, 2].asscalar()))
      #  print("Epoch [{0}] Train loss {1:.4f} val loss {2:.4f} "
            #  .format(epoch, history[epoch, 1].asscalar(), history[epoch, 3].asscalar()))

        # We perform early-stopping regularization, to prevent the model from overfitting
       # df = df.append(df2, ignore_index=True)
        '''
        if new_val_accuracy > best_val_accuracy:
            print('Validation accuracy is increasing..from {1} to {0} - deep copying'
                  .format(new_val_accuracy,best_val_accuracy))
            #model_name_temp = checkpoint_name.format(epoch=epoch, val_acc=new_val_accuracy)  # model_name + "_epoch_" + str(epoch) + ".hdf5"
            #my_net.save_parameters(os.path.join(model_dir, model_name_temp))
            # break
            best_model_name = checkpoint_name.format(epoch=epoch, val_acc=new_val_accuracy)  # model_name + "_epoch_" + str(epoch) + ".hdf5"
            best_model = copy.deepcopy(net) # Keep the best models parameters (Temp)
            #print(best_model_name)
            best_val_accuracy = new_val_accuracy
        else:
            print("NO improvement from the validation accuracy... !!!!")

    # at last: save the best model to HDD
    best_model.save_parameters(os.path.join(model_dir, best_model_name))
    '''
    return history


###########################################################
#  Ploting history and save plots to plots directory
###########################################################
def plot_and_save_training_history(history_ndarray):
    history_data = history_ndarray.as_in_context(mx.cpu()).asnumpy()
    df = pd.DataFrame(history_data, columns=['train_acc', 'train_loss', 'val_acc', 'val_loss'])
    pie = df.plot()
    fig = pie.get_figure()
    fig.savefig(os.path.join(plot_dir, acc_loss_plot_name))


#############################################################


##################################################
#  Start Training
##################################################
print('Start Training....')

history = train_model(my_net, trainer, softmax_cross_entropy, num_epochs=NO_OF_EPOCHS)

print('Training finished..')

##################################################
#  Plotting accuracy and loss
##################################################
print("start plottinng..!!")
plot_and_save_training_history(history)
print("accuracy and loss plot saved to dir =", plot_dir)

print("Training done completly... !!!!")

#######################################################
#  Saving the model
#######################################################

#print('Saving model..')
#my_net.save_parameters(os.path.join(model_dir, model_name))
#print('Model saved to=', model_dir)


#######################################################



