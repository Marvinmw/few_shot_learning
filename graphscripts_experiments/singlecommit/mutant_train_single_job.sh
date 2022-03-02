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


for p in  Closure Math JacksonCore Mockito JxPath Time
do
output_folder=results/single_${p}_intra_mutants_contrastive_${num_class}_loss_both/context
bash mutant_train_single.sh gat "pretrained_models/context/gat/model_0" ${output_folder}/ attention $device $num_class both $p results/mutants_class_contrastive_2_loss_CE/context/gat/saved_model.pt
done

for p in  Closure Math JacksonCore Mockito JxPath Time
do
output_folder=results/single_${p}_intra_mutants_contrastive_${num_class}_loss_CE/context
bash mutant_train_single.sh gat "pretrained_models/context/gat/model_0" ${output_folder}/ attention $device $num_class CE $p results/mutants_class_contrastive_2_loss_CE/context/gat/saved_model.pt
done


for p in  Closure Math JacksonCore Mockito JxPath Time
do
output_folder=results/single_${p}_intra_mutants_scratch_${num_class}_loss_both/context
bash mutant_train_single.sh gat "pretrained_models/context/gat/model_0" ${output_folder}/ attention $device $num_class both $p -1
done

for p in  Closure Math JacksonCore Mockito JxPath Time
do
output_folder=results/single_${p}_intra_mutants_scratch_${num_class}_loss_CE/context
bash mutant_train_single.sh gat "pretrained_models/context/gat/model_0" ${output_folder}/ attention $device $num_class CE $p -1
done

