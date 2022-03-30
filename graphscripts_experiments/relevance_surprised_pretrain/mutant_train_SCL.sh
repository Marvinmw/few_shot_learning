#!/bin/bash -l
#SBATCH -n 3
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH --time=2-00:00:00
#SBATCH --qos=normal
#SBATCH -J scl_r
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=wei.ma@uni.lu
#SBATCH -o %x-%j.log
#SBATCH -C volta32

conda activate graph
device=0
num_class=2
# collections csv io lang text
for test_project in collections csv io lang text
do
for loss in  SCL
do
output_folder=results/transfer_supervised_relevance/${loss}/mutants_${num_class}_loss_${loss}_rm_${test_project}/context
bash run.sh gat "pretrained_models/context/gat/model_0" ${output_folder}/ attention $device $num_class $loss $test_project
done
done


