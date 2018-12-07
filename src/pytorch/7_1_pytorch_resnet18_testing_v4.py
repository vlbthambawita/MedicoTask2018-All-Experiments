#  # Developer: Vajira Thambawita
#  # Last modified date: 18/07/2018
#  # ##################################

#  # Description ##################
#  # pythroch resnet18 testing


from __future__ import print_function, division

import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import os


from sklearn.metrics import confusion_matrix



plt.ion()   # interactive mode


########################################
data_dir = 'data_v2'
model_dir = 'pytorch_models'
model_name = '7_1_resnet18_kavisar_2018_07_17_v7'

#  ##########################################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#  ########################################################################

##################################################
#  Creating the model to load pretrained weights
###################################################

model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 8)

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

model_ft.load_state_dict(torch.load(os.path.join(model_dir, model_name)))
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

test_datasets = datasets.ImageFolder(os.path.join(data_dir, 'test'), test_data_transforms)

test_dataloader = torch.utils.data.DataLoader(test_datasets, batch_size=64, shuffle=False, num_workers=4)

print('Preparing test data finesed')

####################################################
#  # Testing
####################################################
model_ft.eval()
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


print('copying some data back to cpu for generating confusion matrix...')
testset_labels = all_labels_d.to('cpu')
testset_predicted_labels = all_predictions_d.to('cpu')

cm = confusion_matrix(testset_labels, testset_predicted_labels)  # confusion matrix

print('Accuracy of the network on the %d test images: %d %%' % (total, (
        100 * correct / total)))


print(cm)

print('Finished.. ')
