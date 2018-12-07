#!/usr/bin/env bash



model_name_save_and_load="17_1_denseNet201_70_30"

main_dir="../data/data_generated_medicotask_70_30_v2"

training_file="../src/keras/17_1_keras_denseNet_16_classes_medicoDataset_v1_training_v1.py"
retraining_file="../src/keras/17_1_keras_denseNet_16_classes_medicoDataset_v1_re-training_v1.py.py"
# testing_file="../src/keras/16_1_keras_resnet50_15_classes_medicoDataset_v1_testing_v1.py"


gpu_info_plotter="../src/keras/keras_gpu_info_plotting.py"


gpu_monitor_script="gpu_monitor.sh"
gpu_info_dir="keras_gpu_info"


echo "All variable values are OK"

# echo "Exist without doing testing... "
# exit # TODO : remove comment to do only training




function train(){

    echo "Start Training...."

    ./run.sh "$training_file" "$main_dir" "$gpu_info_dir" "training" "$gpu_monitor_script" "$gpu_info_plotter" "$model_name_save_and_load"

    echo "stopped trainning"
    echo "sleeping 2s - relaxing..."
    sleep 2s
}


function retrain(){

    echo "Start Re-Training...."

    ./run.sh "$retraining_file" "$main_dir" "$gpu_info_dir" "re-training" "$gpu_monitor_script" "$gpu_info_plotter" "$model_name_save_and_load"

    echo "stopped re-trainning"
    echo "sleeping 2s - relaxing..."
    sleep 2s
}

function test(){
    echo "strating testing"

    ./run.sh "$testing_file" "$main_dir" "$gpu_info_dir" "testing" "$gpu_monitor_script" "$gpu_info_plotter" "$model_name_save_and_load"


    echo "Testing Finised"

    echo "All done"

}

for var in "$@"
do
    echo "$var"
    $var
done