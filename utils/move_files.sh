#!/usr/bin/env bash

source_dir="../data/data_Medico_2018_development_set_v3_15_classes/normal-z-line"
destination_dir="../data/data_Medico_2018_development_set_v3_15_classes/esophagitis_normal_z_line"

i=1

for file in $source_dir/*.jpg
do
    mv "$file" "${destination_dir}/z_${i}.jpg"
    #echo ${file/_h.png/_half.png}
    ((i++))
done


rm -rf $source_dir