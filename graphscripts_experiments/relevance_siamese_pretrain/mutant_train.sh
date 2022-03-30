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
output_folder=results/transfer_siamese_relevance/mutants_${num_class}_rm_${test_project}/context
bash run.sh gat "pretrained_models/context/gat/model_0" ${output_folder}/ attention $device $num_class $test_project
done


