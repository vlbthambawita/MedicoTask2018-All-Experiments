25/07/2018
===============

TODO 1: Try ResNet-152  lowest top-1 error and top-5 error
TODO 2: Keras ResNet-18 and 152 implementations
TODO 3: Try handling unbalanced data set


FIXME 1:  change no of epochs per step in learning rate scheduler

Keras ResNet-152 implementations
+++++++++++++++++++++++++++++++++++++++++

    local machine
    -------------
changed bs 64 =  memory error
bs = 32 - memory error
bs = 16
epochs per step in learning rate scheduler = 3

    server
    -------

  bs =  64 it is working


 TODO: Resnet152 - 20 epochs - bs = 64








 ##############################################

rsync -avzh mediaEval_2018_structured_v2 --exclude 'mediaEval_2018_structured_v2/data' vajira@10.174.0.52:/home/vajira/myTest/

## Without utils
rsync -avzh mediaEval_2018_structured_v2 --exclude 'mediaEval_2018_structured_v2/data' --exclude 'mediaEval_2018_structured_v2/utils' vajira@10.174.0.52:/home/vajira/myTest/


 ##############################################
