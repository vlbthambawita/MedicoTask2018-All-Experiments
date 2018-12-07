#  # Developer: Vajira Thambawita
#  # Last modified date: 18/07/2018
#  # ##################################

#  # Description ##################
#  # pythroch resnet18 training






###########################################
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
#  import numpy as np
#  import torchvision
from torchvision import datasets, models, transforms, utils



import matplotlib as mpl
#  ######################################################
#  #### Matplotlib X display error - removing for server#
#  ######################################################
mpl.use('Agg')  # This has to run before pyplot import

import matplotlib.pyplot as plt
import time
import os
import copy
import sys
import pandas as pd
import numpy as np

#  from sklearn.metrics import confusion_matrix
import itertools


plt.ion()   # interactive mode

###################################################################
#  Getting main data directory
###############################################################

arg_main_data_dir = sys.argv[1]  # Main data directory to be handled
arg_model_name = sys.argv[2] # model name to be saves
arg_check_point_name_format = sys.argv[3]
arg_mode = sys.argv[4]  # Mode of the runing - Train, Test or Retrain from the best weight file

my_file_name = arg_model_name #"8_1_pytorch_resnet18_v1"  # model name to be saved

#####################################################################

# This is an additional test - Not necessary one

if not arg_mode in ["train", "test", "re-train"]:
    print("Invalid mode type")
    #exit(0)

print("Mode of runing: ", arg_mode)
#exit()

###############################################################

#  Set parameters here

data_dir = arg_main_data_dir
model_dir = data_dir + '/pytorch_models'
plot_dir  = data_dir + '/pytorch_plots'
history_dir = data_dir + '/pytorch_history'


model_name = my_file_name # take my file name as the model name
checkpoint_name_format = arg_check_point_name_format # "13_1_weights-improvement-{epoch:02d}-{val_acc:.4f}.hdf5"
# checkpoint_name = ""

acc_loss_plot_name = 'acc_loss_plot_' + model_name
accuracy_plot_name = 'accuracy_plot_' + model_name
loss_plot_name = 'loss_plot_' + model_name

number_of_epochs = 1

batch_size = 24

########################################################################
#  Managin Directory structure
########################################################################
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

if not os.path.exists(plot_dir):
    os.mkdir(plot_dir)

if not os.path.exists(history_dir):
    os.mkdir(history_dir)


print("1 - Folder structure created")
#####################################################################

#  Preparing Data - Training and Validation + testing
if arg_mode == "train" or "retrain":
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(229),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(90),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        'validation': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(229),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'validation']}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                  shuffle=True, num_workers=1)
                   for x in ['train', 'validation']}


    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'validation']}


##################################################
#  Preparing test data
#################################################

elif arg_mode == "test":

    print('Preparing test data...')

    test_data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_datasets = datasets.ImageFolder(os.path.join(data_dir, 'validation'), test_data_transforms)

    test_dataloader = torch.utils.data.DataLoader(test_datasets, batch_size=batch_size, shuffle=False, num_workers=4)

    print('Preparing test data finised')


###########################################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("2 - Device set to :", device)
#########################################################################

#########################################################################
#  Printing images just for testing
#########################################################################
'''
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

dataiter = iter(dataloaders['train'])
sample_images, sample_labels = dataiter.next()



npimg = sample_images[0].numpy()

npimg = np.transpose(npimg,(1,2,0))



plt.imshow(npimg[:,:, 0])
plt.show()
print(npimg[:, :, 0])
#imshow(utils.make_grid(sample_images))
input()
exit()
'''

#######################################################################


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    history_tensor = torch.empty((num_epochs, 4), device=device)  # 4- trai_acc, train_loss, val_acc, val_loss
    checkpoint_name = None

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)



        # Each epoch has a training and validation phase
        for phase in ['train', 'validation']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            indicator = 0  # just for print batch processing status (no of batches)

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:



                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                  #  print("outputs=", outputs) # only for testing - vajira
                  #  print("labels = ", labels) # only for testing - vajira
                    print(indicator, sep='-', end='=', flush=True)
                    indicator += 1

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # Collecting data for making plots
            if phase == 'train':
                history_tensor[epoch, 0] = epoch_acc
                history_tensor[epoch, 1] = epoch_loss
            if phase == 'validation':
                history_tensor[epoch, 2] = epoch_acc
                history_tensor[epoch, 3] = epoch_loss

            # deep copy the model
            if phase == 'validation' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                checkpoint_name = checkpoint_name_format.format(epoch=epoch, val_acc=best_acc)
                print("Found a best model:", checkpoint_name)
            elif phase== 'validation':
                print("No improvement from the previous best model ")

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, history_tensor, checkpoint_name

##########################################################
#  Model testing method
##########################################################
def test_model(test_model, testDataLoader):
    test_model.eval()
    correct = 0
    total = 0
    all_labels_d = torch.tensor([], dtype=torch.long).to(device)
    all_predictions_d = torch.tensor([], dtype=torch.long).to(device)

    with torch.no_grad():
        for inputs, labels in test_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model_ft(inputs)
            _, predicted = torch.max(outputs.data, 1)
            print((predicted == labels).sum())
            total += labels.size(0)
            correct += (predicted == labels).sum()
            all_labels_d = torch.cat((all_labels_d, labels), 0)
            all_predictions_d = torch.cat((all_predictions_d, predicted), 0)



###########################################################
#  Ploting history and save plots to plots directory
###########################################################
def plot_and_save_training_history(history_tensor):
    history_data = history_tensor.cpu().numpy()
    df = pd.DataFrame(history_data, columns=['train_acc', 'train_loss', 'val_acc', 'val_loss'])
    pie = df.plot()
    fig = pie.get_figure()
    fig.savefig(os.path.join(plot_dir, "_training_" + acc_loss_plot_name))


#############################################################
#  Loading a pretraind model and modifing the last layers

model_ft = models.densenet161(pretrained=False)

#print(model_ft)
#exit()
# num_ftrs = model_ft.fc.in_features
num_ftrs = model_ft.classifier.in_features
model_ft.classifier = nn.Linear(num_ftrs, 16)

#print(model_ft)
#exit() # Just for testing

print("3 - Model created")


## #######################################################
# If multiple GPUS are there, run on multiple GPUS
##########################################################
#  Setting model in multiple GPUs
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    model_ft = nn.DataParallel(model_ft)
elif torch.cuda.device_count() == 1:
    print("Found only one GPU")
else:
    print("No GPU.. Runing on CPU")


print("4 - Setup data parallel devices")

##########################################################
# If the mode == re-training - load the best weight file
# If the mode == test - load the best weight file
##########################################################
if arg_mode == "re-train" or "test":
    best_weight_file_name = input('Please, enter the best weights value file name:')
    model_ft.load_state_dict(torch.load(os.path.join(model_dir, best_weight_file_name)))
    print('4-1 - Model loaded with the best weight file')



##############################################################
#  Loading model to GPUs
##############################################################


# Moving model to the GPu has to be done before setting parameters
# to the model. parameters of the model has different object type
# after moving the model to the GPU (According to the pytorch document)

model_ft = model_ft.to(device)

print("5 - Model loadded to device")

############################################################
# Setting model parameters
############################################################
criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized

optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.1, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)

print("6 - Setup parameters, This has to be done after loading model to the device")
#############################################################
# Start Training
############################################################

model_ft, history_tensor, check_point_name = train_model(model_ft, criterion,
                                                         optimizer_ft, exp_lr_scheduler,
                                                         num_epochs=number_of_epochs)


print("7 - Training complered")
############################################################
# Save the model to the directory
############################################################

if not os.path.exists(model_dir):
    os.mkdir(model_dir)  # to save plots

if not check_point_name==None:
    print(check_point_name)
    torch.save(model_ft.state_dict(), os.path.join(model_dir, check_point_name))
    print("8 -Model saved")
########################################################

plot_and_save_training_history(history_tensor)

print("9 - Plots saved to", plot_dir)


