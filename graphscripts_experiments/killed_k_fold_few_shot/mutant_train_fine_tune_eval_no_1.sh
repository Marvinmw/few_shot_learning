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
#SBATCH -C volta32
conda activate graph
device=0
num_class=2



for train_project in text collections 
do
for fine_tune in no
do
    output_folder=results/few_shot_killed_fine_tune_${fine_tune}/mutants_${num_class}_train_${train_project}_fine_tune_${fine_tune}/context
    bash eval.sh gat "pretrained_models/context/gat/model_0" ${output_folder}/gat/ attention $device $num_class $train_project results/transfer_siamese_killed/mutants_2_rm_${train_project}/context/gat
done
done


