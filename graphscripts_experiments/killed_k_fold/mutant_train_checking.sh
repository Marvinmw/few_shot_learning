#!/bin/bash -l
#SBATCH -n 3
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH --time=2-00:00:00
#SBATCH --qos=normal
#SBATCH -J killed_k_fold_checking
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=wei.ma@uni.lu
#SBATCH -o %x-%j.log
#SBATCH -C volta32
conda activate graph
device=0
num_class=2

for train_project in  collections text lang csv io 
do
for loss in SCL CE
do
output_folder=results/supervised_killed/${loss}/mutants_${num_class}_loss_${loss}_train_${train_project}/context
bash run.sh gat "pretrained_models/context/gat/model_0" ${output_folder}/ attention $device $num_class $loss $train_project
done
done


