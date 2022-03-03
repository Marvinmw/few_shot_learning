#!/bin/bash
gnn_type=$1
pretrainpath=$2
output_prefix=$3
gp=$4
device=$5
num_class=$6
loss=$7
project=$8
saved_model_file=$9
dropratio=0.1
cd ../../

output=${output_prefix}/${gnn_type}
sw=lstm
jk=sum
lstm_emb_dim=150
mkdir -p $output
python mutants_siamese_singlecommit.py --batch_size 256 --num_workers 5  --epochs 20 --num_layer 5 \
--subword_embedding  $sw \
--lstm_emb_dim $lstm_emb_dim \
--graph_pooling $gp \
--JK $jk \
--saved_model_path ${output} \
--log_file ${output}/log.txt \
--gnn_type $gnn_type \
--sub_token_path ./tokens/jars \
--emb_file emb_100.txt \
--dataset DV_PDG \
--input_model_file ${pretrainpath} \
--device ${device} \
--num_class ${num_class} \
--lr 0.001 \
--dropratio $dropratio \
--warmup_schedule yes \
--mutant_type no \
--loss $loss \
--saved_model_file  $saved_model_file \
--lazy yes \
--projects $project


