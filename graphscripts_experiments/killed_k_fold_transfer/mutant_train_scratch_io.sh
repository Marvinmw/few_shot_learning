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

for train_project in  io_12  io_9 csv_27 csv_17 csv_20 lang_21
do
for loss in SCL CE
do
output_folder=results/rerun/supervised_killed_transferweights/${loss}/mutants_${num_class}_loss_${loss}_train_${train_project}/context
supervised_premodel=results/transfer_supervised_killed/CE/mutants_2_loss_CE_rm_${train_project}/context/gat/
bash run.sh gat "pretrained_models/context/gat/model_0" ${output_folder}/ attention $device $num_class $loss $train_project $supervised_premodel
done
done


