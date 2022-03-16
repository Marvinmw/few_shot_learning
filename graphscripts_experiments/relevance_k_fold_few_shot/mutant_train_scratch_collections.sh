#!/bin/bash -l
#SBATCH -n 3
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH --time=2-00:00:00
#SBATCH --qos=normal
#SBATCH -J collections_r
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=wei.ma@uni.lu
#SBATCH -o %x-%j.log

conda activate graph
device=0
num_class=2

for train_project in  collections
do
for fine_tune in yes no
do
    output_folder=results/few_shot_relevance_fine_tune_${fine_tune}/mutants_${num_class}_train_${train_project}_fine_tune_${fine_tune}/context
    bash run.sh gat "pretrained_models/context/gat/model_0" ${output_folder}/ attention $device $num_class $train_project $fine_tune results/transfer_siamese_relevance/mutants_2_rm_${train_project}/context/gat/saved_model.pt
done
done