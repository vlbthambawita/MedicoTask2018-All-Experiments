import os
import mxnet as mx
import numpy as np
import sys
import pandas as pd

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

checkpoint_name = "mxnet-resnet50-weights-improvement-{epoch:02d}-{val_acc:.4f}.hdf5"
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

########################################################
#  New evaluate accuracy method
#  This should be optimized for GPU (making nd array in GPU)
########################################################


def evaluate_accuracy_and_loss(data_iterator, loss_fn, net):
    metric_acc = mx.metric.Accuracy()
    #  numerator = 0.
    #  denominator = 0.
    cumulative_loss = 0.
    no_of_samples = 0

    for i, (data, label) in enumerate(data_iterator):

        with autograd.predict_mode():

            data = data.astype(np.float32).as_in_context(ctx)
            label = label.astype(np.int32).as_in_context(ctx)
            output = net(data)
            loss = loss_fn(output, label)
            prediction = nd.argmax(output, axis=1).astype(np.int32)
            cumulative_loss += nd.sum(loss).asscalar()
            no_of_samples += data.shape[0]

        metric_acc.update([label], [prediction])
        # metric_loss.update([label], [prediction])

    print("cumulative loss = {0} no_of_samples = {1}".format(cumulative_loss, no_of_samples))
    loss = cumulative_loss / no_of_samples

    return (metric_acc.get()[1], loss)


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


acc_before, loss_before = evaluate_accuracy_and_loss(dataloader_test, softmax_cross_entropy, my_net)

#print("Untrained network Test Accuracy: {0:.4f}".format(evaluate_accuracy_gluon(dataloader_test, my_net)))

print("Untrained network Test Accuracy: {0:.4f} Test Loss: {1:.4f} ".format(acc_before, loss_before))



##################################################
##################################################
#  Training
##################################################
##################################################

def train_model(net, trainer, loss_fn, num_epochs=1):
    best_val_accuracy = 0
    df = pd.DataFrame(columns=['train_acc', 'train_loss', 'val_acc', 'val_loss'])

    for epoch in range(num_epochs):
        for i, (data, label) in enumerate(dataloader_train):
            data = data.astype(np.float32).as_in_context(ctx)
            label = label.as_in_context(ctx)

            if i % 20 == 0 and i > 0:
                print('Batch [{0}] loss: {1:.4f}'
                      .format(i, loss.mean().asscalar()))

            with autograd.record():
                output = net(data)
                loss = softmax_cross_entropy(output, label)
                loss.backward()
            trainer.step(data.shape[0])

        nd.waitall()  # wait at the end of the epoch

        train_accuracy, train_loss = evaluate_accuracy_and_loss(dataloader_train, loss_fn, net)
        new_val_accuracy, new_val_loss = evaluate_accuracy_and_loss(dataloader_test, loss_fn, net)
        df2 = pd.DataFrame([[train_accuracy, train_loss, new_val_accuracy, new_val_loss]],
                           columns=['train_acc', 'train_loss', 'val_acc', 'val_loss'])
        # new_val_accuracy = evaluate_accuracy_gluon(dataloader_test, my_net)
        print("all done")
        print(type(train_accuracy))
        print("Epoch [{0}] Train accuracy {1:.4f} val Accuracy {2:.4f} "
              .format(epoch, train_accuracy, new_val_accuracy))
        print("Epoch [{0}] Train loss {1:.4f} val loss {2:.4f} "
              .format(epoch, train_loss, new_val_loss))

        # We perform early-stopping regularization, to prevent the model from overfitting
        df = df.append(df2, ignore_index=True)
        if new_val_accuracy > best_val_accuracy:
            print('Validation accuracy is increasing.. saving the model')
            model_name_temp = checkpoint_name.format(epoch=epoch, val_acc=new_val_accuracy)  # model_name + "_epoch_" + str(epoch) + ".hdf5"
            my_net.save_parameters(os.path.join(model_dir, model_name_temp))
            # break
            best_val_accuracy = new_val_accuracy
        else:
            print("NO improvement from the validation accuracy... !!!!")


    return df


###########################################################
#  Ploting history and save plots to plots directory
###########################################################
def plot_and_save_training_history(history_df):

    df = pd.DataFrame(history_df, columns=['train_acc', 'train_loss', 'val_acc', 'val_loss'])
    pie = df.plot()
    fig = pie.get_figure()
    fig.savefig(os.path.join(plot_dir, acc_loss_plot_name))


#############################################################


##################################################
#  Start Training
##################################################
print('Start Training....')

history_df = train_model(my_net, trainer, softmax_cross_entropy, num_epochs=NO_OF_EPOCHS)

print('Training finished..')

##################################################
#  Plotting accuracy and loss
##################################################

plot_and_save_training_history(history_df)
print("accuracy and loss plot saved to dir =", plot_dir)

print("Training done completly... !!!!")

#######################################################
#  Saving the model
#######################################################

#print('Saving model..')
#my_net.save_parameters(os.path.join(model_dir, model_name))
#print('Model saved to=', model_dir)


#######################################################


