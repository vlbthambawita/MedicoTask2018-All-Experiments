#!/usr/bin/env bash

#######################################################
DATE_WITH_TIME=`date "+%Y%m%d-%H%M%S"`
########################################################
echo $DATE_WITH_TIME
#######################################################
model_name_save_and_load="$DATE_WITH_TIME:24_6_1_medico_v1"
check_point_name_format="$DATE_WITH_TIME:24_6_1_weights-improvement-{epoch:02d}-{val_acc:.4f}.hdf5"


main_dir="../data/data_generated_medicotask_70_30_modified_v3"

########################################################

training_file="../src/pytorch/24_6_pytorch_densenet161_resnet152_averaging_medioDataset_TEST_RESULT_GENERATOR_v2.py"
retraining_file="$training_file" # "../src/pytorch/13_6_pytorch_densenet161_medioDataset_v1_training_v1.py"
testing_file="$training_file" # "../src/pytorch/13_6_pytorch_densenet161_medioDataset_v1_training_v1.py"

gpu_info_plotter="../src/pytorch/pytorch_gpu_info_plotting.py"

gpu_monitor_script="gpu_monitor.sh"
gpu_info_dir="pytorch_gpu_info"


##############################################################################
echo "All variable values are OK"



###########################################################

function train(){

    echo "Start Training...."

    ./run2.sh "$training_file" "$main_dir" "$gpu_info_dir" "train" "$gpu_monitor_script" "$gpu_info_plotter" "$model_name_save_and_load" "$check_point_name_format"

    echo "stopped trainning"

    echo "sleeping 2s - relaxing..."
    sleep 2s

    echo "Training completed "

}

##############################################################

function retrain(){

    echo "Start Re-Training...."

    ./run2.sh "$retraining_file" "$main_dir" "$gpu_info_dir" "re-train" "$gpu_monitor_script" "$gpu_info_plotter" "$model_name_save_and_load" "$check_point_name_format"

    echo "stopped re-trainning"
    echo "sleeping 2s - relaxing..."
    sleep 2s
}

##############################################################

function test(){


    echo "strating testing"

    ./run2.sh "$testing_file" "$main_dir" "$gpu_info_dir" "test" "$gpu_monitor_script" "$gpu_info_plotter" "$model_name_save_and_load"


    echo "Testing Finised"

}

###################################################

for var in "$@"
do
    echo "$var"
    $var
done
#####################################################

echo "All done"