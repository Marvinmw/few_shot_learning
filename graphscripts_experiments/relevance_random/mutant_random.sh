#!/bin/bash -l
#SBATCH -n 5
#SBATCH -N 1
#SBATCH -p batch
#SBATCH --time=2-00:00:00
#SBATCH --qos=normal
#SBATCH -J ce_r
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=wei.ma@uni.lu
#SBATCH -o %x-%j.log

conda activate graph
num_class=2
# collections csv io lang text
for test_project in collections csv io lang text
do
output_folder=results/random_prior/mutants_${num_class}_rm_${test_project}/context
bash run.sh ${output_folder}/ $num_class $test_project
done


