#!/bin/bash -l
#SBATCH -n 3
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH --time=2-00:00:00
#SBATCH --qos=normal
#SBATCH -J io_killed
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=wei.ma@uni.lu
#SBATCH -o %x-%j.log

conda activate graph
device=0
num_class=2

for train_project in  io
do
for fine_tune in yes no
do
    output_folder=results/few_shot_killed_fine_tune_${fine_tune}/mutants_${num_class}_train_${train_project}_fine_tune_${fine_tune}/context
    load_model=results/transfer_siamese_killed/mutants_2_rm_${train_project}/context/gat/
    bash run.sh gat "pretrained_models/context/gat/model_0" ${output_folder}/ attention $device $num_class $train_project $fine_tune $load_model
done
done

