#  # Developer: Vajira Thambawita
#  # Last modified date: 18/07/2018
#  # ##################################

#  # Description ##################
#  # pythroch resnet18 testing


from __future__ import print_function, division

import torch
import torch.nn as nn
import numpy as np

from torchvision import datasets, models, transforms

#  ######################################################
#  #### Matplotlib X display error - removing for server#
#  ######################################################
import matplotlib as mpl
mpl.use('Agg')  # This has to run before pyplot import


import matplotlib.pyplot as plt
import os
import sys


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import itertools


plt.ion()   # interactive mode


########################################
###################################################################
#  Getting main data directory
###############################################################

main_data_dir = sys.argv[1]  # Main data directory to be handled

model_name = sys.argv[2] # model name to be loaded

best_weight_file_name = input('Please, enter the best weights value file name:')

my_file_name = model_name  # Model name to be loaded



data_dir = main_data_dir
model_dir = data_dir + '/pytorch_models'
plot_dir  = data_dir + '/pytorch_plots'

model_name = my_file_name

cm_plot_name = 'cm_'+model_name

batch_size = 32

#########################################
#  Managing Directory
#########################################
if not os.path.exists(plot_dir):
    os.mkdir(plot_dir)



#  ##########################################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#  ########################################################################

##################################################
#  Creating the model to load pretrained weights
###################################################

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

print('Model architecture created..')


#####################################################
#  Setting model in multiple GPUs
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    model_ft = nn.DataParallel(model_ft)
elif torch.cuda.device_count() == 1:
    print("Found only one GPU")
else:
    print("No GPU.. Runing on CPU")

##############################################3

# Loading the saved model for testing

model_ft.load_state_dict(torch.load(os.path.join(model_dir, best_weight_file_name)))
print('Model loaded')
model_ft = model_ft.to(device)

print('Model loaded... and set to device')

print('Started to test on testing data...')


##################################################
#  Preparing test data
#################################################

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

####################################################
#  # Testing
####################################################
model_ft.eval()
correct = 0
total = 0
all_labels_d = torch.tensor([], dtype=torch.float).to(device)
all_predictions_d = torch.tensor([], dtype=torch.float).to(device)

with torch.no_grad():
    for inputs, labels in test_dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model_ft(inputs)
        #_, predicted = torch.max(outputs.data, 1)
        predicted = outputs.round()
        predicted = predicted.view(labels.size())
        predicted = predicted.float()
        labels = labels.type(torch.cuda.FloatTensor)
        print((predicted == labels).sum())
        total += labels.size(0)
        correct += (predicted == labels).sum()
        all_labels_d = torch.cat((all_labels_d, labels), 0)
        all_predictions_d = torch.cat((all_predictions_d, predicted), 0)


print('copying some data back to cpu for generating confusion matrix...')
testset_labels = all_labels_d.cpu()
testset_predicted_labels = all_predictions_d.cpu()   # to('cpu')

cm = confusion_matrix(testset_labels, testset_predicted_labels)  # confusion matrix

print('Accuracy of the network on the %d test images: %f %%' % (total, (
        100.0 * correct / total)))


print(cm)

################################################################
#  Plotting Confusion Matrix
################################################################

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

print("taking class names to plot CM")

class_names = test_datasets.classes  # taking class names for plotting confusion matrix

print("Generating confution matrix")

plot_confusion_matrix(cm, classes=class_names, title='my confusion matrix')

print('confusion matrix saved to ', plot_dir)

##################################################################
# classification report
#################################################################
print(classification_report(testset_labels, testset_predicted_labels, target_names=class_names))

print('Finished.. ')
