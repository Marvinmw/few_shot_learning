#!/bin/bash
output_prefix=$1
num_class=$2
test_project=$3

cd ../../

output=${output_prefix}/
mkdir -p $output
python dumpy_classifier.py \
--saved_model_path ${output} \
--log_file ${output}/log.txt \
--dataset DV_PDG \
--num_class ${num_class} \
--task relevance \
--test_projects $test_project \
--projects collections csv io lang text 


