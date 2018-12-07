#  # Developer: Vajira Thambawita
#  # Last modified date: 18/07/2018
#  # ##################################

#  # Description ##################
#  # pythroch resnet18 training






###########################################
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
#  import numpy as np
#  import torchvision
from torchvision import datasets, models, transforms




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

#  from sklearn.metrics import confusion_matrix
#  import itertools


plt.ion()   # interactive mode

###################################################################
#  Getting main data directory
###############################################################

arg_main_data_dir = sys.argv[1]  # Main data directory to be handled
arg_model_name = sys.argv[2] # model name to be saves
arg_check_point_name_format = sys.argv[3]

my_file_name = arg_model_name #"8_1_pytorch_resnet18_v1"  # model name to be saved

###########################################

#  Set parameters here
data_dir = arg_main_data_dir
model_dir = data_dir + '/pytorch_models'
plot_dir  = data_dir + '/pytorch_plots'
history_dir = data_dir + '/pytorch_history'


model_name = my_file_name # take my file name as the model name
checkpoint_name_format = arg_check_point_name_format#"14_1_weights-improvement-{epoch:02d}-{val_acc:.4f}.hdf5"
# checkpoint_name = ""

acc_loss_plot_name = 'acc_loss_plot_' + model_name
accuracy_plot_name = 'accuracy_plot_' + model_name
loss_plot_name = 'loss_plot_' + model_name

number_of_epochs = 10

batch_size = 32

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

#  Preparing Data - Training and Validation

data_transforms = {
    'train': transforms.Compose([
       # transforms.RandomResizedCrop(224),
        transforms.Resize(35),
        transforms.CenterCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(90),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'validation': transforms.Compose([
        transforms.Resize(35),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'validation']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                              shuffle=True, num_workers=4)
               for x in ['train', 'validation']}


dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'validation']}


#  ##########################################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#  ########################################################################

def train_model(model, criterion, optimizer, scheduler, num_epochs=25): # remove scheduler
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
               #  scheduler.step()
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
                    new_output =  outputs.view(labels.size())
                    #_, preds = torch.max(outputs, 1)
                    preds= outputs.round()
                    preds = preds.view(labels.size())
                    labels = labels.type(torch.cuda.FloatTensor)
                    loss = criterion(new_output, labels)
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

model_ft = models.resnet50(pretrained=True)

#print(model_ft)
#exit()
num_ftrs = model_ft.fc.in_features
#num_ftrs = model_ft.classifier.in_features
fc_layers = nn.Sequential(
               # nn.Linear(num_ftrs, 4096),
               # nn.AvgPool2d(3, stride=2), #inbuild average pooling is there
               # nn.Dropout(p=0.2),
                nn.Linear(num_ftrs, 1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, 1),
                nn.Sigmoid()
                #nn.Dropout(p=0.1),
            )

# model_ft.fc = nn.Linear(num_ftrs, 1)
model_ft.fc = fc_layers

#print(model_ft)
# exit()
#print(model_ft)
#exit() # Just for testing

####################################################################
####################################################################
#  My own nerual network from scratch
####################################################################
####################################################################


class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.conv1= nn.Conv2d(3, 6, 5)
        self.pool1= nn.MaxPool2d(2,2)
        self.conv2= nn.Conv2d(6,16,5)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(16 * 5 * 5, 120) # planing to pass through 2 max pooling layers
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1) # for binary classifications

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x



myModel1 = MyNet()
####################################################################
############################################################
#  # Setting model parameters
#criterion = nn.CrossEntropyLoss()
#criterion = nn.BCEWithLogitsLoss()
criterion = nn.BCELoss()

# Observe that all parameters are being optimized
#optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
# optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.01, weight_decay=1e-6, momentum=0.9, nesterov=True)
optimizer_ft = optim.SGD(myModel1.parameters(), lr=0.01,  momentum=0.9)
#optimizer_ft = optim.Adadelta(model_ft.parameters(), rho=0.95)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)

## #######################################################
# If multiple GPUS are there, run on multiple GPUS
##########################################################
#  Setting model in multiple GPUs
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
   # model_ft = nn.DataParallel(model_ft)
    myModel1 = nn.DataParallel(myModel1)
elif torch.cuda.device_count() == 1:
    print("Found only one GPU")
else:
    print("No GPU.. Runing on CPU")

##############################################################
#  Loading model to GPUs and setting parameters
##############################################################
#model_ft = model_ft.to(device)
myModel1 = myModel1.to(device)


#  setting grad to TRue
#for param in model_ft.parameters():
 #   param.requires_grad = True

#############################################################
### start Training
############################################################

myModel1, history_tensor, check_point_name = train_model(myModel1, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=number_of_epochs)

############################################################
### Save the model to the directory
############################################################

if not os.path.exists(model_dir):
    os.mkdir(model_dir)  # to save plots

if not check_point_name==None:
    print(check_point_name)
    torch.save(myModel1.state_dict(), os.path.join(model_dir, check_point_name))
    print("Model saved")
########################################################

plot_and_save_training_history(history_tensor)

print("Plots saved to", plot_dir)
