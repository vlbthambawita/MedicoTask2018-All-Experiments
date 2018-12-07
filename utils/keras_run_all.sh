#!/usr/bin/env bash

main_dir="../data/data_generated_medicotask_v1"

gpu_info_dir="keras_gpu_info"

model_name_save_and_load="11_1_keras_resnet50_2_runs_5_epc_selecting_best_model_v4"


gpu_monitor_script="8_1_gpu_monitor.sh"
training_file="../src/keras/11_1_keras_resnet50_fully_optimized_medicoDataset_v1_training_v1.py"
testing_file="../src/keras/11_1_keras_resnet50_fully_optimized_medicoDataset_v1_testing_v1.py"
gpu_info_plotter="../src/keras/keras_gpu_info_plotting.py"

echo "Start Training...."

./8_1_run.sh "$training_file" "$main_dir" "$gpu_info_dir" "training" "$gpu_monitor_script" "$gpu_info_plotter" "$model_name_save_and_load"

echo "stopped trainning"

echo "sleeping 2s - relaxing..."
sleep 2s

# echo "Exist without doing testing... "
# exit # TODO : remove comment to do only training

echo "strating testing"

./8_1_run.sh "$testing_file" "$main_dir" "$gpu_info_dir" "testing" "$gpu_monitor_script" "$gpu_info_plotter" "$model_name_save_and_load"


echo "Testing Finised"

echo "All done"


function train(){

    echo "Training fuction is working"
}