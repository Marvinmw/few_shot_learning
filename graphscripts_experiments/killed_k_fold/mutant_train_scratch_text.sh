#!/bin/bash -l
#SBATCH -n 3
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH --time=2-00:00:00
#SBATCH --qos=normal
#SBATCH -J text_killed
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=wei.ma@uni.lu
#SBATCH -o %x-%j.log

conda activate graph
device=0
num_class=2

for train_project in  text
do
for loss in SCL CE
do
output_folder=results/supervised_killed/${loss}/mutants_${num_class}_loss_${loss}_train_${train_project}/context
supervide_premodel=results/transfer_supervised_killed/CE/mutants_2_loss_CE_rm_${train_project}/context/gat/saved_model.pt
bash run.sh gat "pretrained_models/context/gat/model_0" ${output_folder}/ attention $device $num_class $loss $train_project $supervide_premodel
done
done


