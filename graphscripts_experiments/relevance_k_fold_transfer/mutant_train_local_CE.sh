#!/bin/bash -l
#SBATCH -n 3
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH --time=2-00:00:00
#SBATCH --qos=normal
#SBATCH -J jsc
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=wei.ma@uni.lu
#SBATCH -o %x-%j.log
#SBATCH -C volta32

conda activate graph
device=0
num_class=2
# collections csv io lang text


for train_project in   lang 
do
for loss in CE
do
output_folder=results/supervised/${loss}/mutants_relevance_${num_class}_loss_${loss}_train_${train_project}/context
bash run.sh gat "pretrained_models/context/gat/model_0" ${output_folder}/ attention $device $num_class $loss $train_project results/transfer_supervised_relevance/CE/mutants_2_loss_CE_rm_lang/context/gat/saved_model.pt
done
done


