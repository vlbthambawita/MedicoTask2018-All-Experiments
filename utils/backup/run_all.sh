#!/usr/bin/env bash

main_dir="data_v2"
gpu_info_dir="pytorch_gpu_info"
training_file="7_1_pytorch_resnet18_training_v6.py"
testing_file="7_1_pytorch_resnet18_testing_v6.py"

echo "Start Training...."

run.sh "$training_file" "$main_dir" "$gpu_info_dir" "training"

echo "stopped trainning"

echo "sleeping 2s - relaxing..."
sleep 2s

echo "strating testing"

run.sh "$testing_file" "$main_dir" "$gpu_info_dir" "testing"


echo "Testing Finised"

echo "All done"
