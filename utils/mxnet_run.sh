#!/usr/bin/env bash

#!/usr/bin/env bash



model_name_save_and_load="3_1_mxnet_resnet50_medicotask_v1"

training_file="../src/mxnet/3_1_mxnet_resnet50_medicoTask_training_v2.py"
retraining_file="../src/mxnet/2_1_mxnet_resnet50_medicoTask_re-training_v1.py"
testing_file="../src/mxnet/3_1_mxnet_resnet50_medicoTask_testing_v1.py"


gpu_info_plotter="../src/mxnet/mxnet_gpu_info_plotting.py"

main_dir="../data/data_generated_medicotask_v1"
gpu_monitor_script="gpu_monitor.sh"
gpu_info_dir="mxnet_gpu_info"


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