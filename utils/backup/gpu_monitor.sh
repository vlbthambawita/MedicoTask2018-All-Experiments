#!/usr/bin/env bash

echo "gpu deamon started ....."

file_name=$1
main_data_dir=$2

gpu_info_directory_name="$3"

full_path="$main_data_dir/$gpu_info_directory_name/$file_name"


mkdir -p "$main_data_dir"/"$gpu_info_directory_name" ## Make directory if not exist to save GPU information

rm -f "$full_path" ## if file is exist, then remove it

echo "start monitoring gpu usage......."

nvidia-smi --query-gpu=\
index,\
timestamp,\
utilization.gpu,\
utilization.memory,\
temperature.gpu,\
power.draw,\
clocks.sm,\
clocks.mem,\
clocks.gr\
 --format=csv,nounits -l 1 >> $full_path