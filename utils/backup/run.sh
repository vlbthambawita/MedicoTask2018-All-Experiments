#!/usr/bin/env bash

#### Get first input argument as the file name
python_script=$1
gpu_file_name=$1.csv
main_dir=$2
gpu_info_dir=$3
train_or_test=$4

gpu_info_file_full_path="$main_dir/$gpu_info_dir/$gpu_file_name"

echo "Starting a deamon process to collect gpu information....."


screen -dmS gpu_win ./gpu_monitor.sh "$gpu_file_name" "$main_dir" "$gpu_info_dir"

echo "waiting 2s to start python script..."
sleep 2

echo "Starting python script..."
python3 "$python_script"
echo "Python script completed"


echo "waiting 2s to stop gpu information collector..."
sleep 2

screen -S gpu_win -X quit

echo "starting gpu information plotting.."
sleep 2


#python3 0_gpu_info_plotting.py $gpu_info_file_full_path $train_or_test
python3 0_7_gpu_info_plotting_v2.py $gpu_info_file_full_path $train_or_test $python_script

echo "Test OK"
